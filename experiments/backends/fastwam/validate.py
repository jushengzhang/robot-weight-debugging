"""Backend validation: override exactness, cross-process determinism, eager-vs-compiled, record, frames, probe."""
import json, sys, glob, numpy as np
from pathlib import Path
sys.path.insert(0, "/mnt/SSD2_16TB/jusheng/wsp/fastwam")
import client as C
F = Path("/mnt/SSD2_16TB/jusheng/wsp/fastwam"); R = F / "runs/validate"
socks = [s for s in C.list_socks() if "/g3_" in s][-2:]  # two GPU-3 servers
eps = [json.loads(l) for l in (F / "samples/determinism5.jsonl").read_text().splitlines()]
A = {(r["suite"], r["variant_id"]): r for r in map(json.loads, open(F / "runs/determinism_A/results.jsonl"))}
key = lambda r: (r["suite"], r["variant_id"])
log = lambda m: None
out = {}
# 1 identity override (orig,edit paired): edit must equal orig; orig must equal run A (other GPU / process)
rs = C.rollout(eps, R / "identity", socks, weights_override=F / "overrides/test_identity", arms=["orig", "edit"], log=log)
o = {key(r): r for r in rs if r["arm"] == "orig"}; e = {key(r): r for r in rs if r["arm"] == "edit"}
out["cross_process_match_vs_runA"] = sum(o[k]["action_sha1"] == A[k]["action_sha1"] for k in o)
out["identity_edit_eq_orig"] = sum(o[k]["action_sha1"] == e[k]["action_sha1"] for k in o)
out["identity_restore_verified"] = all(r.get("restore_verified") for r in e.values())
# 2 destructive overrides: must change behaviour; afterwards orig must be bit-identical again
for name in ["test_zero_head", "test_noise_l20"]:
    rs = C.rollout(eps, R / name, socks, weights_override=F / f"overrides/{name}", arms=["edit"], log=log)
    out[f"{name}_changed_actions"] = sum(r["action_sha1"] != A[key(r)]["action_sha1"] for r in rs)
    out[f"{name}_success"] = f'{sum(r["success"] for r in rs)}/{len(rs)} (orig {sum(A[key(r)]["success"] for r in rs)}/{len(rs)})'
    out[f"{name}_restore_verified"] = all(r.get("restore_verified") for r in rs)
rs = C.rollout(eps, R / "orig_after_edits", socks, log=log)
out["orig_after_edits_match_runA"] = sum(r["action_sha1"] == A[key(r)]["action_sha1"] for r in rs)
# 3 eager (record) run with activations + frames
rec = {"action_blocks": [0, 15, 29], "video_blocks": [15, 29], "what": ["attn_out", "mlp_out", "resid"]}
rs = C.rollout(eps[:2], R / "record", socks, record=rec, save_frames=50, log=log)
out["eager_vs_compiled_action_match"] = sum(r["action_sha1"] == A[key(r)]["action_sha1"] for r in rs)
out["eager_vs_compiled_success_match"] = sum(r["success"] == A[key(r)]["success"] for r in rs)
z = np.load(rs[0]["activations"]); out["act_keys_example"] = sorted(z.files)[:6]; out["n_act_arrays"] = len(z.files)
out["act_shapes"] = {k: list(z[k].shape) for k in sorted(z.files) if k.startswith("mean::")}
out["frames"] = rs[0].get("frames_steps")
# 4 probe on saved inputs (orig+edit), activations + attention
inp = sorted(glob.glob(rs[0]["frames_dir"] + "/*_inputs.pt"))[:2]
p = C.probe(inp, R / "probe", socks[0], weights_override=F / "overrides/test_noise_l20", arms=["orig", "edit"],
            record={"action_blocks": [29], "video_blocks": [29], "what": ["resid"]}, attention_layers=[0, 15, 29])
res = p["results"]
# probe(orig) on the inputs saved at step 0 must reproduce the chunk executed in the rollout
import torch
saved = torch.load(inp[0], weights_only=False)["action_chunk"]
out["probe_reproduces_rollout_chunk"] = bool(np.allclose(np.load(res[0]["action"]), saved, atol=0))
out["probe_edit_differs"] = float(np.abs(np.load(res[0]["action_norm"]) - np.load(res[len(inp)]["action_norm"])).max())
a = np.load(res[0]["attention"]["15"] if "15" in res[0]["attention"] else res[0]["attention"][15])
out["attn_shape"] = list(a.shape); out["attn_rowsum_to_video"] = float(a.sum(-1).mean())
out["probe_restore"] = p["override"]
print(json.dumps(out, indent=1, default=str))
(R / "validation.json").write_text(json.dumps(out, indent=1, default=str))
