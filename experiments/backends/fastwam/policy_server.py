#!/usr/bin/env python
"""FastWAM (uncond, LIBERO 2cam224) policy server for LIBERO-Plus: rollouts with weight overrides,
activation capture, replayable policy-input dumps, and a simulation-free PROBE (actions/activations/attention).

One process = one model copy on one GPU. Unix socket; one JSON line request -> one JSON line response;
requests are served sequentially. See README.md for the request format.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import logging
import os
import random
import re
import socket
import sys
import time
import traceback
from pathlib import Path

W = "/mnt/SSD2_16TB/jusheng/Documents/Codex/2026-10-02/vla-wam-cvpr-gemini-api-key/work"
FASTWAM_ROOT = Path(W) / "baselines/FastWAM"
WSP = Path("/mnt/SSD2_16TB/jusheng/wsp/fastwam")

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("LIBERO_CONFIG_PATH", str(WSP / "libero_cfg"))
os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(Path(W) / "checkpoints/FastWAM-common"))
os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(WSP / "cache/inductor"))

for p in (str(FASTWAM_ROOT), str(FASTWAM_ROOT / "experiments/libero")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from PIL import Image  # noqa: E402

import eval_libero_single as up  # noqa: E402  (registers OmegaConf resolvers eval/max/split)
import fastwam.models.wan22.mot as mot_mod  # noqa: E402
from experiments.libero.libero_utils import (  # noqa: E402
    LIBERO_ENV_RESOLUTION,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    invert_gripper_action,
)
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT  # noqa: E402
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json  # noqa: E402
from fastwam.utils.pytorch_utils import set_global_seed  # noqa: E402

with contextlib.redirect_stdout(io.StringIO()):
    from libero.libero import benchmark  # noqa: E402

LOG = logging.getLogger("policy_server")
CLASSIFICATION = WSP / "third_party/LIBERO-plus/libero/libero/benchmark/task_classification.json"
EXPERTS = {"action": "action_expert", "video": "video_expert"}
WHAT = {"attn_out": "self_attn.o", "mlp_out": "ffn", "resid": "gate"}  # resid = 2nd gate call (block output)


class Variants:
    def __init__(self):
        self.meta = {}
        for suite, items in json.loads(CLASSIFICATION.read_text()).items():
            for it in items:
                self.meta[(suite, int(it["id"]))] = it
        self._suites = {}

    def suite(self, name):
        if name not in self._suites:
            with contextlib.redirect_stdout(io.StringIO()):  # suppress huge "[info] using task orders"
                self._suites[name] = benchmark.get_benchmark_dict()[name](task_order_index=0)
        return self._suites[name]


# ----------------------------------------------------------------------------------------------
# Activation recorder
# ----------------------------------------------------------------------------------------------
class Recorder:
    """Spec (either form, can be combined):
      {"action_blocks": [i,...], "video_blocks": [j,...], "what": ["attn_out","mlp_out","resid"], "per_call": false}
      {"modules": ["action_expert.blocks.3.ffn.0", ...]}   (raw module names; '<...>.blocks.<i>.out' = resid)
    Per hooked call the output [B, L, D] is averaged over tokens:
      action expert -> key '<name>::action'      (mean over the action tokens)
      video expert  -> keys '<name>::agent', '<name>::wrist' (mean over first-frame video tokens of each camera;
                      the 2 cameras are concatenated horizontally, agent = left half of the latent grid)
    Episode/probe summary = mean over all hooked calls (all policy queries x denoising steps for the action
    expert; one prefill pass per query for the video expert)."""

    def __init__(self, server, spec):
        self.server = server
        named = dict(server.model.named_modules())
        self.per_call = bool(spec.get("per_call", False))
        targets = []  # (key, module_name, is_resid, expert)
        for exp_short, field in (("action", "action_blocks"), ("video", "video_blocks")):
            for i in spec.get(field, []) or []:
                for w in spec.get("what", ["resid"]):
                    targets.append((f"{exp_short}.{int(i)}.{w}", f"{EXPERTS[exp_short]}.blocks.{int(i)}.{WHAT[w]}",
                                    w == "resid", exp_short))
        for name in spec.get("modules", []) or []:
            resid = name.endswith(".out")
            mname = name[:-4] + ".gate" if resid else name
            exp = "video" if (".video" in mname or mname.startswith("video")) else "action"
            targets.append((name, mname, resid, exp))
        self.handles = []
        for key, mname, resid, exp in targets:
            if mname not in named:
                raise KeyError(f"record: unknown module '{mname}' (from '{key}')")
            self.handles.append(named[mname].register_forward_hook(self._hook(key, resid, exp)))
        self.reset()

    def _hook(self, key, resid, exp):
        def hook(_m, _i, out):
            if resid:
                k = self.parity.get(key, 0)
                self.parity[key] = k + 1
                if k % 2 == 0:
                    return
            t = out[0] if isinstance(out, (tuple, list)) else out
            if not torch.is_tensor(t) or t.ndim != 3:
                return
            t = t.detach().float()
            if exp == "video":
                f, h, w = self.server.video_grid
                if t.shape[1] == f * h * w:
                    g = t[0].reshape(f, h, w, -1)[0]  # first (only) frame in action inference
                    vals = {"agent": g[:, : w // 2].reshape(-1, g.shape[-1]).mean(0),
                            "wrist": g[:, w // 2:].reshape(-1, g.shape[-1]).mean(0)}
                else:
                    vals = {"all": t.reshape(-1, t.shape[-1]).mean(0)}
            else:
                vals = {"action": t.reshape(-1, t.shape[-1]).mean(0)}
            for sub, v in vals.items():
                kk = f"{key}::{sub}"
                self.sum[kk] = self.sum[kk] + v if kk in self.sum else v.clone()
                self.cnt[kk] = self.cnt.get(kk, 0) + 1
                if self.per_call:
                    self.calls.setdefault(kk, []).append(v.cpu())

        return hook

    def reset(self):
        self.sum, self.cnt, self.calls, self.parity = {}, {}, {}, {}

    def summary(self):
        out = {f"mean::{k}": (s / self.cnt[k]).cpu().numpy() for k, s in self.sum.items()}
        out.update({f"ncalls::{k}": np.array(c) for k, c in self.cnt.items()})
        if self.per_call:
            out.update({f"calls::{k}": torch.stack(v).numpy() for k, v in self.calls.items()})
        return out

    def dump(self, path):
        np.savez(path, **self.summary())
        return {k: int(v) for k, v in self.cnt.items()}

    def close(self):
        for h in self.handles:
            h.remove()


class AttnCapture:
    """Patches mot.flash_attention (eager only) to also compute action->video attention probabilities
    for the chosen layers during the action-expert pass. Result per layer: [heads, L_action, L_video],
    averaged over denoising steps."""

    def __init__(self, layers, num_layers):
        self.layers = set(int(x) for x in layers)
        self.num_layers = num_layers
        self.orig = mot_mod.flash_attention
        self.acc, self.n, self.counter = {}, {}, 0

    def __enter__(self):
        orig = self.orig

        def wrapped(q, k, v, num_heads, ctx_mask=None, **kw):
            out = orig(q, k, v, num_heads, ctx_mask=ctx_mask, **kw)
            if q.shape[1] < k.shape[1]:  # action pass (keys = video cache + action tokens)
                layer = self.counter % self.num_layers
                self.counter += 1
                if layer in self.layers:
                    B, Lq, HD = q.shape
                    d = HD // num_heads
                    qh = q.float().view(B, Lq, num_heads, d).transpose(1, 2)
                    kh = k.float().view(B, k.shape[1], num_heads, d).transpose(1, 2)
                    s = qh @ kh.transpose(-1, -2) / (d ** 0.5)
                    if ctx_mask is not None:
                        m = ctx_mask.to(s.device)
                        s = s.masked_fill(~m, float("-inf")) if m.dtype == torch.bool else s + m.float()
                    p = s.softmax(-1)[0]  # [H, Lq, Lk]
                    lv = k.shape[1] - Lq
                    pv = p[:, :, :lv].cpu()
                    self.acc[layer] = self.acc[layer] + pv if layer in self.acc else pv
                    self.n[layer] = self.n.get(layer, 0) + 1
            return out

        mot_mod.flash_attention = wrapped
        return self

    def __exit__(self, *a):
        mot_mod.flash_attention = self.orig

    def result(self):
        return {l: (self.acc[l] / self.n[l]).numpy() for l in sorted(self.acc)}


_SUFFIX = re.compile(r"\s+(?:(?:view|tb|table|light|noise|add|language|initstate)\s+-?\d+|level\d+|sample\d+)(?:\s.*)?$")


def strip_variant_suffix(s: str) -> str:
    """'put the bowl on the plate view 342 15 100 0 0 initstate 0' -> 'put the bowl on the plate'"""
    return _SUFFIX.sub("", s).strip()


def sha256_tensor(t: torch.Tensor) -> str:
    c = t.detach().contiguous().cpu()
    return hashlib.sha256(c.view(torch.uint8).numpy().tobytes()).hexdigest()


# ----------------------------------------------------------------------------------------------
# Server
# ----------------------------------------------------------------------------------------------
class PolicyServer:
    language_mode = "strip"  # default: task.language with the LIBERO-Plus variant suffix removed (see README)

    def __init__(self, args):
        self.args = args
        self.device = "cuda"
        overrides = [f"ckpt={args.ckpt}", "model.redirect_common_files=false",
                     f"EVALUATION.dataset_stats_path={args.stats}"]
        with initialize_config_dir(version_base="1.3", config_dir=str(FASTWAM_ROOT / "configs")):
            self.cfg = compose(config_name="sim_libero.yaml", overrides=overrides)
        cfg = self.cfg
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)
        t0 = time.time()
        dtype = up._mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
        self.model = instantiate(cfg.model, model_dtype=dtype, device=self.device)
        self.model.load_checkpoint(str(args.ckpt))
        self.model = self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.processor = instantiate(cfg.data.train.processor).eval()
        self.processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(args.stats)))
        self.action_horizon = int(cfg.data.train.num_frames) - 1
        self.input_h, self.input_w = [int(x) for x in cfg.data.train.video_size]
        self.num_inference_steps = int(cfg.EVALUATION.num_inference_steps)
        self.load_time = time.time() - t0
        # all aliases: video_expert.* == mot.mixtures.video.* == dit.mixtures.video.* (same tensors)
        self.params = dict(self.model.named_parameters(remove_duplicate=False))
        self.backup = {}
        self.variants = Variants()
        # capture the latent grid of the video tokens (for per-camera activation split)
        self.video_grid = (1, 0, 0)
        orig_prepare = self.model.video_expert.prepare

        def prepare(*a, **k):
            out = orig_prepare(*a, **k)
            self.video_grid = (int(out[6]), int(out[7]), int(out[8]))
            return out

        self.model.video_expert.prepare = prepare
        LOG.info("model loaded in %.1fs", self.load_time)

    # ---------------- info ----------------
    def info(self):
        groups = {}
        for n, p in self.model.named_parameters():  # deduplicated
            top = n.split(".")[0]
            groups[top] = groups.get(top, 0) + p.numel()
        return {
            "top_level_param_counts": groups,
            "total_params": sum(groups.values()),
            "gpu_mem_allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "gpu_mem_reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "gpu_max_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
            "load_time_s": self.load_time, "pid": os.getpid(),
            "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }

    def list_params(self, prefix=""):
        return [[n, list(p.shape), str(p.dtype).replace("torch.", "")]
                for n, p in self.model.named_parameters() if n.startswith(prefix)]

    # ---------------- weight override ----------------
    def resolve_name(self, name):
        for cand in (name, "mot." + name, name.removeprefix("model.")):
            if cand in self.params:
                return cand
        raise KeyError(f"override: unknown parameter '{name}'")

    def _load_override(self, d):
        d = Path(d)
        tensors = {}
        for f in (sorted(d.iterdir()) if d.is_dir() else [d]):
            if f.suffix == ".safetensors":
                from safetensors.torch import load_file
                tensors.update(load_file(str(f)))
            elif f.suffix == ".npy":
                tensors[f.stem] = torch.from_numpy(np.load(f))
            elif f.suffix == ".npz":
                z = np.load(f)
                tensors.update({k: torch.from_numpy(z[k]) for k in z.files})
        if not tensors:
            raise ValueError(f"override: no tensors found in {d}")
        return tensors

    def apply_override(self, d, mode="replace", applied=None):
        tensors = self._load_override(d)
        applied = [] if applied is None else applied
        with torch.no_grad():
            for raw, t in tensors.items():
                name = self.resolve_name(raw)
                p = self.params[name]
                if tuple(t.shape) != tuple(p.shape):
                    raise ValueError(f"override: shape mismatch for {raw}: {tuple(t.shape)} vs {tuple(p.shape)}")
                key = id(p)
                if key not in self.backup:
                    cpu = p.detach().cpu().clone()
                    self.backup[key] = (name, cpu, sha256_tensor(cpu))
                orig = self.backup[key][1].to(self.device).float()
                new = t.to(self.device, dtype=torch.float32)
                new = orig + new if mode == "add" else orig * new if mode == "scale" else new
                p.data.copy_(new.to(p.dtype))  # in place: keeps data_ptr (valid under CUDA graphs)
                applied.append(key)
        return applied

    def restore(self, keys):
        ok = True
        h = hashlib.sha256()
        with torch.no_grad():
            for key in keys:
                name, cpu, digest = self.backup[key]
                p = self.params[name]
                p.data.copy_(cpu.to(self.device))
                ok &= bool(torch.equal(p.detach().cpu(), cpu) and sha256_tensor(p) == digest)
                h.update(digest.encode())
        return ok, h.hexdigest()

    @contextlib.contextmanager
    def arm(self, arm, override, mode):
        if arm == "orig":
            yield None
            return
        if not override:
            raise ValueError("arm 'edit' needs weights_override")
        keys, info = [], {}
        try:
            self.apply_override(override, mode, keys)
            info["n_params"] = len(set(keys))
            yield info
        finally:
            ok, digest = self.restore(keys)
            info.update(restore_verified=ok, orig_sha256_digest=digest)
            if not ok:
                raise RuntimeError("weight restore verification FAILED")

    # ---------------- inference ----------------
    def infer(self, inputs, compile_infer):
        kw = dict(prompt=inputs["prompt"], input_image=inputs["input_image"].to(self.device, self.model.torch_dtype),
                  action_horizon=int(inputs.get("action_horizon", self.action_horizon)),
                  negative_prompt="", text_cfg_scale=1.0,
                  num_inference_steps=int(inputs.get("num_inference_steps", self.num_inference_steps)),
                  proprio=inputs["proprio"], sigma_shift=None, seed=int(inputs["seed"]),
                  rand_device="cpu", tiled=False, compile_action_infer=bool(compile_infer))
        with torch.no_grad():
            return self.model.infer_action(**kw)["action"]  # [T, D] normalized, float32 cpu

    def postprocess(self, action_norm):
        """identical to upstream _predict_action_chunk post-processing"""
        action = up._denormalize_action(action_norm, self.processor)[0]
        action[..., -1] = action[..., -1] * 2 - 1
        action = invert_gripper_action(action)
        if bool(self.cfg.EVALUATION.get("binarize_gripper", False)):
            action[..., -1] = np.sign(action[..., -1])
        return action

    def make_inputs(self, obs, task_description, seed):
        image, proprio, imgs = up._obs_to_model_input(
            obs, cfg=self.cfg, processor=self.processor, width=self.input_w, height=self.input_h,
            device=self.device, dtype=self.model.torch_dtype)
        return {"prompt": DEFAULT_PROMPT.format(task=task_description), "input_image": image,
                "proprio": proprio, "seed": int(seed), "action_horizon": self.action_horizon,
                "num_inference_steps": self.num_inference_steps}, imgs

    # ---------------- episodes ----------------
    @staticmethod
    def _seed_all(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def run_episode(self, ep, *, compile_infer, recorder, out_dir, save_video, save_frames, tag):
        suite_name = ep["suite"]
        seed = int(ep.get("seed", 7))
        init_idx = int(ep.get("init_idx", 0))
        vid = int(ep["variant_id"]) if "variant_id" in ep else int(ep["task_idx"]) + 1
        task_idx = vid - 1
        meta = self.variants.meta.get((suite_name, vid), {})
        suite = self.variants.suite(suite_name)
        task = suite.get_task(task_idx)
        res = {"suite": suite_name, "variant_id": vid, "seed": seed, "init_idx": init_idx, "arm": tag,
               "name": task.name, "category": meta.get("category"), "difficulty": meta.get("difficulty_level"),
               "success": False, "steps": 0, "failure": None}
        stem = f"{suite_name}_v{vid}_s{seed}_i{init_idx}_{tag}"
        t0 = time.time()
        env = None
        try:
            self._seed_all(seed)
            init_states = suite.get_task_init_states(task_idx)
            for attempt in range(5):
                try:
                    env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, seed)
                    break
                except Exception as exc:
                    if exc.__class__.__name__ != "RandomizationError" or attempt == 4:
                        raise
            res["instruction_raw"] = task_description
            clean = getattr(env, "language_instruction", None)
            if isinstance(clean, (list, tuple)):
                clean = " ".join(clean)
            res["instruction_clean"] = clean
            mode = ep.get("language", self.language_mode)
            if mode == "clean" and clean:
                task_description = str(clean).strip('"')
            elif mode == "strip":
                task_description = strip_variant_suffix(task_description)
            res["instruction"] = task_description
            max_steps = int(ep.get("max_steps", up._get_max_steps(suite_name)))
            wait = int(self.cfg.EVALUATION.num_steps_wait)
            replan = int(self.cfg.EVALUATION.replan_steps)
            frame_every = int(save_frames) if save_frames else 0
            fdir = None
            if frame_every and out_dir:
                fdir = Path(out_dir) / f"{stem}_frames"
                fdir.mkdir(parents=True, exist_ok=True)
            env.reset()
            obs = env.set_init_state(init_states[init_idx % len(init_states)])
            if recorder is not None:
                recorder.reset()
            frames, pending, done, t, n_chunks, trace, saved = [], [], False, 0, 0, [], []
            while t < max_steps + wait:
                if t < wait:
                    obs, _, done, _ = env.step(get_libero_dummy_action())
                    t += 1
                    continue
                st = t - wait
                if not pending:
                    inputs, imgs = self.make_inputs(obs, task_description, seed)
                    chunk = self.postprocess(self.infer(inputs, compile_infer))
                    n_chunks += 1
                    pending = chunk[:replan].tolist()
                    if fdir is not None and st % frame_every == 0:
                        Image.fromarray(np.concatenate([imgs["image"], imgs["wrist_image"]], 1)).save(fdir / f"t{st:04d}.png")
                        torch.save({**{k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in inputs.items()},
                                    "suite": suite_name, "variant_id": vid, "step": st, "arm": tag,
                                    "action_chunk": chunk}, fdir / f"t{st:04d}_inputs.pt")
                        saved.append(st)
                else:
                    imgs = get_libero_image(obs) if save_video else None
                if save_video and imgs is not None and st % 2 == 0:
                    frames.append(np.concatenate([imgs["image"], imgs["wrist_image"]], axis=1))
                a = pending.pop(0)
                trace.append(a)
                obs, _, done, _ = env.step(a)
                if done:
                    break
                t += 1
            res["success"] = bool(done)
            res["steps"] = int(t - wait + (1 if done else 0))
            res["n_queries"] = n_chunks
            res["termination"] = "success" if done else "max_steps"
            if not done:
                res["failure"] = "timeout"
            ah = np.asarray(trace, dtype=np.float32)
            res["action_sha1"] = hashlib.sha1(np.round(ah, 5).tobytes()).hexdigest()[:16]
            if out_dir is not None:
                np.save(Path(out_dir) / f"{stem}_actions.npy", ah)
                res["actions"] = str(Path(out_dir) / f"{stem}_actions.npy")
            if fdir is not None:
                res["frames_dir"] = str(fdir)
                res["frames_steps"] = saved
            if recorder is not None and out_dir is not None:
                p = Path(out_dir) / f"{stem}_acts.npz"
                res["activation_calls"] = recorder.dump(p)
                res["activations"] = str(p)
            if save_video and out_dir is not None and frames:
                import imageio
                vp = Path(out_dir) / f"{stem}_{'succ' if done else 'fail'}.mp4"
                imageio.mimwrite(str(vp), frames, fps=10)
                res["video"] = str(vp)
        except Exception as exc:
            res["failure"] = f"error: {exc!r}"
            res["termination"] = "error"
            res["traceback"] = traceback.format_exc()[-3000:]
        finally:
            if env is not None:
                with contextlib.suppress(Exception):
                    env.close()
        res["wall_s"] = round(time.time() - t0, 2)
        return res

    def _common(self, req):
        out_dir = req.get("out_dir")
        if out_dir:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
        override = req.get("weights_override")
        arms = req.get("arms") or (["edit"] if override else ["orig"])
        record = req.get("record")
        compile_infer = bool(req.get("compile", record is None))
        if record is not None and compile_infer:
            raise ValueError("record requires compile=false (hooks do not fire inside CUDA-graph replays)")
        return out_dir, override, arms, record, compile_infer

    def run(self, req):
        out_dir, override, arms, record, compile_infer = self._common(req)
        self.language_mode = req.get("language", "strip")
        recorder = Recorder(self, record) if record else None
        results, override_info = [], None
        try:
            for ep in req["episodes"]:
                for arm in arms:
                    with self.arm(arm, override, req.get("override_mode", "replace")) as info:
                        r = self.run_episode(ep, compile_infer=compile_infer, recorder=recorder, out_dir=out_dir,
                                             save_video=bool(req.get("save_video", False)),
                                             save_frames=req.get("save_frames", 0), tag=arm)
                    if info is not None:
                        override_info = info
                        r["restore_verified"] = info["restore_verified"]
                    results.append(r)
        finally:
            if recorder is not None:
                recorder.close()
        return {"ok": True, "results": results, "override": override_info,
                "gpu_max_reserved_gb": torch.cuda.max_memory_reserved() / 1e9}

    def probe(self, req):
        """inputs: list of paths to *_inputs.pt (from save_frames). For each arm and input: predicted action
        chunk (normalized + executed/denormalized), optional activations (record spec) and optional attention
        maps (attention: {"layers": [...]} <=4 layers, <=4 inputs)."""
        out_dir, override, arms, record, compile_infer = self._common(req)
        out_dir = Path(out_dir or (WSP / "runs/probe_tmp"))
        out_dir.mkdir(parents=True, exist_ok=True)
        attn = req.get("attention")
        if attn and (len(attn.get("layers", [])) > 4 or len(req["inputs"]) > 4):
            raise ValueError("attention maps limited to <=4 inputs and <=4 layers")
        if attn and compile_infer:
            raise ValueError("attention requires compile=false")
        recorder = Recorder(self, record) if record else None
        results, override_info = [], None
        try:
            for arm in arms:
                with self.arm(arm, override, req.get("override_mode", "replace")) as info:
                    for i, path in enumerate(req["inputs"]):
                        inputs = torch.load(path, map_location="cpu", weights_only=False)
                        if "seed_override" in req:
                            inputs["seed"] = int(req["seed_override"])
                        if recorder is not None:
                            recorder.reset()
                        cap = AttnCapture(attn["layers"], self.model.mot.num_layers) if attn else None
                        with (cap if cap is not None else contextlib.nullcontext()):
                            a_norm = self.infer(inputs, compile_infer)
                        r = {"arm": arm, "input": path, "idx": i}
                        stem = out_dir / f"probe_{arm}_{i}"
                        np.save(f"{stem}_action_norm.npy", a_norm.numpy())
                        np.save(f"{stem}_action.npy", self.postprocess(a_norm.clone()))
                        r["action_norm"] = f"{stem}_action_norm.npy"
                        r["action"] = f"{stem}_action.npy"
                        if recorder is not None:
                            r["activations"] = f"{stem}_acts.npz"
                            recorder.dump(r["activations"])
                        if cap is not None:
                            for l, m in cap.result().items():
                                np.save(f"{stem}_attn_L{l}.npy", m)  # [heads, L_action, L_video]
                            r["attention"] = {int(l): f"{stem}_attn_L{l}.npy" for l in cap.result()}
                            r["video_grid_fhw"] = list(self.video_grid)
                        results.append(r)
                if info is not None:
                    override_info = info
        finally:
            if recorder is not None:
                recorder.close()
        return {"ok": True, "results": results, "override": override_info}

    # ---------------- socket loop ----------------
    def serve(self, sock_path):
        sock_path = Path(sock_path)
        sock_path.parent.mkdir(parents=True, exist_ok=True)
        sock_path.unlink(missing_ok=True)
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(str(sock_path))
        srv.listen(8)
        (sock_path.parent / (sock_path.stem + ".ready")).write_text(json.dumps(self.info()))
        LOG.info("listening on %s", sock_path)
        while True:
            conn, _ = srv.accept()
            with conn:
                f = conn.makefile("rwb")
                line = f.readline()
                if not line:
                    continue
                stop = False
                try:
                    req = json.loads(line)
                    cmd = req.get("cmd", "run")
                    if cmd == "ping":
                        resp = {"ok": True, "pid": os.getpid()}
                    elif cmd == "info":
                        resp = {"ok": True, **self.info()}
                    elif cmd == "params":
                        resp = {"ok": True, "params": self.list_params(req.get("prefix", ""))}
                    elif cmd == "run":
                        resp = self.run(req)
                    elif cmd == "probe":
                        resp = self.probe(req)
                    elif cmd == "shutdown":
                        resp, stop = {"ok": True}, True
                    else:
                        resp = {"ok": False, "error": f"unknown cmd {cmd}"}
                except Exception as exc:
                    resp = {"ok": False, "error": repr(exc), "traceback": traceback.format_exc()[-3000:]}
                try:
                    f.write((json.dumps(resp, default=float) + "\n").encode())
                    f.flush()
                except BrokenPipeError:
                    LOG.warning("client went away")
                if stop:
                    break
        srv.close()
        sock_path.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sock", required=True)
    ap.add_argument("--ckpt", default=str(WSP / "ckpt/libero_uncond_2cam224.pt"))
    ap.add_argument("--stats", default=str(WSP / "ckpt/libero_uncond_2cam224_dataset_stats.json"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    PolicyServer(args).serve(args.sock)


if __name__ == "__main__":
    main()
