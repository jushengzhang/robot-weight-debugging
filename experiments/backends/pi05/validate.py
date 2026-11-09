"""Backend validation for the pi0.5 server: determinism (within/across processes and GPUs), override exactness,
destructive edits + exact restore, record, save_frames + probe (replay), attention maps.
usage: venv/bin/python validate.py SOCK_A SOCK_B   (two servers, ideally on different GPUs)"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file

F = Path(__file__).resolve().parent
sys.path.insert(0, str(F))
import client as C  # noqa: E402

R = F / "runs/validate"
socks = sys.argv[1:3]
eps = [json.loads(l) for l in (F / "samples/determinism5.jsonl").read_text().splitlines() if l.strip()]
key = lambda r: (r["suite"], r["variant_id"])  # noqa: E731
log = lambda m: None  # noqa: E731
out = {}

# overrides
od = F / "overrides"
names = {n: (s, d) for n, s, d in C.call(socks[0], {"cmd": "params"})["params"]}
head = "action_out_proj.weight"
l10 = "paligemma_with_expert.gemma_expert.model.layers.10.mlp.down_proj.weight"
ident = od / "test_identity"; zero = od / "test_zero_head"; noise = od / "test_noise_expert_l10"
for d in (ident, zero, noise):
    d.mkdir(parents=True, exist_ok=True)
from safetensors.torch import load_file  # noqa: E402
flat = F / "weights/pi05_libero_flat.safetensors"
from safetensors import safe_open  # noqa: E402
with safe_open(str(flat), "pt") as f:
    w_head = f.get_tensor(head).float().numpy()
    w_l10 = f.get_tensor(l10).float().numpy()
    w_vlm = f.get_tensor("paligemma_with_expert.paligemma.model.language_model.layers.5.self_attn.o_proj.weight").float().numpy()
np.save(ident / f"{head}.npy", w_head)
np.save(ident / "vlm.layers.5.self_attn.o_proj.weight.npy", w_vlm)  # alias form
np.save(zero / f"{head}.npy", np.zeros_like(w_head))
rng = np.random.default_rng(0)
np.save(noise / f"{l10}.npy", (w_l10 + 0.5 * w_l10.std() * rng.standard_normal(w_l10.shape)).astype(np.float32))

# 0 determinism: same eps twice on server A, once on server B
A = {key(r): r for r in C.rollout(eps, R / "detA", [socks[0]], log=log)}
A2 = {key(r): r for r in C.rollout(eps, R / "detA2", [socks[0]], log=log)}
B = {key(r): r for r in C.rollout(eps, R / "detB", [socks[1]], log=log)}
out["determinism_same_process"] = f"{sum(A[k]['action_sha1'] == A2[k]['action_sha1'] for k in A)}/{len(A)}"
out["determinism_cross_process"] = f"{sum(A[k]['action_sha1'] == B[k]['action_sha1'] for k in A)}/{len(A)}"
out["runA"] = {f"{k[0]}:{k[1]}": [A[k]["success"], A[k]["steps"], A[k]["action_sha1"], A[k]["wall_s"]] for k in A}
# 1 identity override (paired orig/edit)
rs = C.rollout(eps, R / "identity", socks[:1], weights_override=ident, arms=["orig", "edit"], log=log)
o = {key(r): r for r in rs if r["arm"] == "orig"}; e = {key(r): r for r in rs if r["arm"] == "edit"}
out["identity_edit_eq_orig"] = f"{sum(o[k]['action_sha1'] == e[k]['action_sha1'] for k in o)}/{len(o)}"
out["identity_restore_verified"] = all(r.get("restore_verified") for r in e.values())
# 2 destructive overrides, then orig must be bit-identical to run A
for d in (zero, noise):
    rs = C.rollout(eps, R / d.name, socks[:1], weights_override=d, arms=["edit"], log=log)
    out[f"{d.name}_changed_actions"] = f"{sum(r['action_sha1'] != A[key(r)]['action_sha1'] for r in rs)}/{len(rs)}"
    out[f"{d.name}_success"] = f"{sum(r['success'] for r in rs)}/{len(rs)} (orig {sum(A[key(r)]['success'] for r in rs)}/{len(rs)})"
    out[f"{d.name}_restore_verified"] = all(r.get("restore_verified") for r in rs)
rs = C.rollout(eps, R / "orig_after_edits", socks[:1], log=log)
out["orig_after_edits_match_runA"] = f"{sum(r['action_sha1'] == A[key(r)]['action_sha1'] for r in rs)}/{len(rs)}"
# 3 record + frames (eager, same numerics as default rollouts)
rec = {"vlm_layers": [0, 9, 17], "expert_layers": [0, 9, 17], "vision_layers": [26], "what": ["attn_out", "mlp_out", "resid"]}
rs = C.rollout(eps[:2], R / "record", socks[:1], record=rec, save_frames=50, log=log)
out["record_run_action_match_runA"] = f"{sum(r['action_sha1'] == A[key(r)]['action_sha1'] for r in rs)}/{len(rs)}"
z = np.load(rs[0]["activations"])
out["n_act_arrays"] = len(z.files)
out["act_shapes"] = {k: list(z[k].shape) for k in sorted(z.files) if k.startswith("mean::")}
out["act_ncalls"] = rs[0]["activation_calls"]
out["frames"] = rs[0].get("frames_steps")
# 4 probe saved inputs (orig + edit), activations + attention
inp = sorted(glob.glob(rs[0]["frames_dir"] + "/*_inputs.pt"))[:2]
p = C.probe(inp, R / "probe", socks[0], weights_override=noise, arms=["orig", "edit"],
            record={"expert_layers": [17], "vlm_layers": [17], "what": ["resid"]}, attention_layers=[0, 9, 17])
res = p["results"]
out["probe_reproduces_rollout_chunk"] = [r["matches_saved_chunk"] for r in res if r["arm"] == "orig"]
out["probe_edit_max_abs_diff"] = float(np.abs(np.load(res[0]["action"]) - np.load(res[len(inp)]["action"])).max())
a = np.load(list(res[0]["attention"].values())[1])
out["attn_shape"] = list(a.shape)
out["attn_prefix_layout"] = res[0].get("prefix_layout")
out["attn_mass_on_prefix"] = float(a.sum(-1).mean())
lay = res[0]["prefix_layout"]
out["attn_mass_base_wrist_rwrist_lang"] = [float(a[:, :, s:e].sum(-1).mean()) for s, e in
                                           (lay["base"], lay["wrist"], lay["rwrist(masked)"], lay["lang"])]
out["probe_override"] = p["override"]
print(json.dumps(out, indent=1, default=str))
(R / "validation.json").write_text(json.dumps(out, indent=1, default=str))
