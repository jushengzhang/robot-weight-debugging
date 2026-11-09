#!/usr/bin/env python
"""pi0.5 (openpi `pi05_libero`, PyTorch port) policy server for LIBERO-Plus: rollouts with weight overrides,
activation capture, replayable policy-input dumps and a simulation-free PROBE (actions / activations / attention).

Mirrors wsp/fastwam/policy_server.py (host 120): one process = one model copy on one GPU, unix socket,
one JSON line request -> one JSON line response, requests served sequentially. See README.md.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import io
import json
import logging
import math
import os
import random
import re
import socket
import sys
import time
import traceback
from pathlib import Path

WSP = Path(os.environ.get("WSP_PI05", str(Path.home() / "wsp_pi05")))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("LIBERO_CONFIG_PATH", str(WSP / "libero_cfg"))
os.environ.setdefault("OPENPI_DATA_HOME", str(WSP / "cache/openpi"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(WSP / "cache/inductor"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

with contextlib.redirect_stdout(io.StringIO()):
    from libero.libero import benchmark, get_libero_path  # noqa: E402
    from libero.libero.envs import OffScreenRenderEnv  # noqa: E402
from openpi_client import image_tools  # noqa: E402
from openpi.policies import policy_config as _policy_config  # noqa: E402
from openpi.training import config as _config  # noqa: E402

LOG = logging.getLogger("pi05_server")
CLASSIFICATION = WSP / "third_party/LIBERO-plus/libero/libero/benchmark/task_classification.json"

# ---- upstream openpi examples/libero/main.py settings ----
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
RESIZE = 224
REPLAN_STEPS = 5
NUM_STEPS_WAIT = 10
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}

# ---- module naming ----
VLM = "paligemma_with_expert.paligemma.model.language_model"
EXPERT = "paligemma_with_expert.gemma_expert.model"
VISION = "paligemma_with_expert.paligemma.model.vision_tower.vision_model"
PROJ = "paligemma_with_expert.paligemma.model.multi_modal_projector"
ALIASES = {"vlm.": VLM + ".", "expert.": EXPERT + ".", "vision.": VISION + ".", "projector.": PROJ + "."}
LAYER_PATH = {"vlm": VLM + ".layers", "expert": EXPERT + ".layers", "vision": VISION + ".encoder.layers"}
WHAT = {"vlm": {"attn_out": "self_attn", "mlp_out": "mlp", "resid": ""},
        "expert": {"attn_out": "self_attn", "mlp_out": "mlp", "resid": ""},
        "vision": {"attn_out": "self_attn", "mlp_out": "mlp", "resid": ""}}
N_IMG_TOK = 256  # 224/14 = 16x16 SigLIP patches per image
IMG_NAMES = ["base", "wrist", "rwrist"]  # base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb (zeros, masked for pi05)


def alias(name: str) -> str:
    for a, full in ALIASES.items():
        if name.startswith(a):
            return full + name[len(a):]
    return name


class Variants:
    def __init__(self):
        self.meta = {}
        if os.environ.get("WSP_ORIG"):  # original LIBERO: no LIBERO-Plus classification
            self._suites = {}
            return
        for suite, items in json.loads(CLASSIFICATION.read_text()).items():
            for it in items:
                self.meta[(suite, int(it["id"]))] = it
        self._suites = {}

    def suite(self, name):
        if name not in self._suites:
            with contextlib.redirect_stdout(io.StringIO()):
                self._suites[name] = benchmark.get_benchmark_dict()[name](task_order_index=0)
        return self._suites[name]


# ----------------------------------------------------------------------------------------------
# Activation recorder
# ----------------------------------------------------------------------------------------------
class Recorder:
    """Spec: {"vlm_layers": [...], "expert_layers": [...], "vision_layers": [...],
              "what": ["attn_out","mlp_out","resid"], "per_call": false}  and/or  {"modules": [raw module names]}
    Hook outputs ([B, L, D]) are averaged over token groups:
      vlm (prefix pass, once per query)  -> '::base', '::wrist' (256 image tokens each), '::lang' (valid prompt tokens)
      expert (10 denoising steps/query)  -> '::action' (the 10 action tokens)
      vision (SigLIP, once per image)    -> '::base', '::wrist' (256 patch tokens; the zero right-wrist image is skipped)
    attn_out = self_attn output (before o-residual; o_proj applied), mlp_out = mlp output, resid = decoder-layer
    output (residual stream after the layer). Episode / probe summary = mean over all hooked calls."""

    def __init__(self, server, spec):
        self.server = server
        named = dict(server.model.named_modules())
        self.per_call = bool(spec.get("per_call", False))
        targets = []
        for grp in ("vlm", "expert", "vision"):
            for i in spec.get(f"{grp}_layers", []) or []:
                for w in spec.get("what", ["resid"]):
                    sub = WHAT[grp][w]
                    mname = f"{LAYER_PATH[grp]}.{int(i)}" + (f".{sub}" if sub else "")
                    targets.append((f"{grp}.{int(i)}.{w}", mname, grp))
        for name in spec.get("modules", []) or []:
            mname = alias(name)
            grp = "vision" if mname.startswith(VISION) else "expert" if mname.startswith(EXPERT) else "vlm"
            targets.append((name, mname, grp))
        self.handles = []
        for key, mname, grp in targets:
            if mname not in named:
                raise KeyError(f"record: unknown module '{mname}' (from '{key}')")
            self.handles.append(named[mname].register_forward_hook(self._hook(key, grp)))
        self.vision_call = {}
        self.reset()

    def _hook(self, key, grp):
        def hook(_m, _i, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if not torch.is_tensor(t) or t.ndim != 3:
                return
            t = t.detach().float()[0]  # [L, D]
            vals = {}
            if grp == "expert":
                vals["action"] = t.mean(0)
            elif grp == "vision":
                k = self.vision_call.get(key, 0)
                self.vision_call[key] = k + 1
                idx = k % 3
                if idx == 2:
                    return
                vals[IMG_NAMES[idx]] = t.mean(0)
            else:
                pm = self.server.last_prefix_mask
                if t.shape[0] == 3 * N_IMG_TOK + (0 if pm is None else pm.numel() - 3 * N_IMG_TOK) and pm is not None:
                    vals["base"] = t[:N_IMG_TOK].mean(0)
                    vals["wrist"] = t[N_IMG_TOK: 2 * N_IMG_TOK].mean(0)
                    lm = pm[3 * N_IMG_TOK:].to(t.device)
                    vals["lang"] = t[3 * N_IMG_TOK:][lm].mean(0)
                else:
                    vals["all"] = t.mean(0)
            for sub, v in vals.items():
                kk = f"{key}::{sub}"
                self.sum[kk] = self.sum[kk] + v if kk in self.sum else v.clone()
                self.cnt[kk] = self.cnt.get(kk, 0) + 1
                if self.per_call:
                    self.calls.setdefault(kk, []).append(v.cpu())

        return hook

    def reset(self):
        self.sum, self.cnt, self.calls, self.vision_call = {}, {}, {}, {}

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
    """Action-expert attention probabilities (eager attention returns them): per chosen expert layer,
    softmax weights of the 10 action queries over the prefix keys (3x256 image tokens + 200 prompt tokens),
    averaged over the 10 denoising steps -> [8 heads, 10 action tokens, L_prefix]."""

    def __init__(self, model, layers):
        named = dict(model.named_modules())
        self.acc, self.n, self.handles = {}, {}, []
        for l in layers:
            m = named[f"{LAYER_PATH['expert']}.{int(l)}.self_attn"]
            self.handles.append(m.register_forward_hook(self._hook(int(l))))

    def _hook(self, l):
        def hook(_m, _i, out):
            w = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else None
            if w is None:
                return
            p = w.detach().float()[0]  # [H, Lq, Lk]
            lq = p.shape[1]
            pv = p[:, :, : p.shape[2] - lq].cpu()
            self.acc[l] = self.acc[l] + pv if l in self.acc else pv
            self.n[l] = self.n.get(l, 0) + 1
        return hook

    def __enter__(self):
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()

    def result(self):
        return {l: (self.acc[l] / self.n[l]).numpy() for l in sorted(self.acc)}


_SUFFIX = re.compile(r"\s+(?:(?:view|tb|table|light|noise|add|language|initstate)\s+-?\d+|level\d+|sample\d+)(?:\s.*)?$")


def strip_variant_suffix(s: str) -> str:
    """'put the bowl on the plate view 342 15 100 0 0 initstate 0' -> 'put the bowl on the plate'"""
    return _SUFFIX.sub("", s).strip()


def sha256_tensor(t: torch.Tensor) -> str:
    c = t.detach().contiguous().cpu()
    return hashlib.sha256(c.view(torch.uint8).numpy().tobytes()).hexdigest()


def quat2axisangle(quat):
    """copied from robosuite (as in openpi examples/libero/main.py)"""
    quat = np.array(quat, dtype=np.float64)
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


# ----------------------------------------------------------------------------------------------
# Server
# ----------------------------------------------------------------------------------------------
class PolicyServer:
    def __init__(self, args):
        self.args = args
        self.device = "cuda"
        t0 = time.time()
        cfg = _config.get_config("pi05_libero")
        cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None))
        self.train_cfg = cfg
        if args.random_init:  # DEBUG ONLY: pipeline test without weights (random init, dummy norm stats)
            from openpi.models_pytorch import pi0_pytorch
            from openpi.shared import normalize as _normalize
            orig_load = type(cfg.model).load_pytorch

            def load_pytorch(self_, train_config, weight_path):
                return pi0_pytorch.PI0Pytorch(config=self_)

            type(cfg.model).load_pytorch = load_pytorch
            ns = _normalize.load(Path(args.ckpt) / "assets/physical-intelligence/libero") \
                if (Path(args.ckpt) / "assets/physical-intelligence/libero/norm_stats.json").exists() else None
            dummy = WSP / "ckpt/_random_init"
            dummy.mkdir(parents=True, exist_ok=True)
            (dummy / "model.safetensors").touch()
            self.policy = _policy_config.create_trained_policy(cfg, dummy, pytorch_device=self.device, norm_stats=ns)
            type(cfg.model).load_pytorch = orig_load
        else:
            self.policy = _policy_config.create_trained_policy(cfg, args.ckpt, pytorch_device=self.device)
        self.model = self.policy._model  # PI0Pytorch
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.eager_sample = self.model.sample_actions
        self.compiled_sample = None
        self.action_horizon = int(cfg.model.action_horizon)
        self.action_dim = int(cfg.model.action_dim)
        self.load_time = time.time() - t0
        self.params = dict(self.model.named_parameters(remove_duplicate=False))
        self.backup = {}
        self.variants = Variants()
        # stash the prefix pad mask of the current query (for per-region activation means)
        self.last_prefix_mask = None
        orig_embed_prefix = self.model.embed_prefix

        def embed_prefix(*a, **k):
            out = orig_embed_prefix(*a, **k)
            self.last_prefix_mask = out[1][0].detach()
            return out

        self.model.embed_prefix = embed_prefix
        LOG.info("model loaded in %.1fs", self.load_time)

    # ---------------- info ----------------
    def info(self):
        groups = {}
        for n, p in self.model.named_parameters():
            top = ".".join(n.split(".")[:4]) if n.startswith("paligemma_with_expert") else n.split(".")[0]
            groups[top] = groups.get(top, 0) + p.numel()
        return {
            "top_level_param_counts": groups, "total_params": sum(groups.values()),
            "gpu_mem_allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "gpu_mem_reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "gpu_max_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
            "load_time_s": self.load_time, "pid": os.getpid(), "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "action_horizon": self.action_horizon, "replan_steps": REPLAN_STEPS, "num_steps_wait": NUM_STEPS_WAIT,
            "max_steps": MAX_STEPS,
        }

    def list_params(self, prefix=""):
        prefix = alias(prefix)
        return [[n, list(p.shape), str(p.dtype).replace("torch.", "")]
                for n, p in self.model.named_parameters() if n.startswith(prefix)]

    # ---------------- weight override ----------------
    def resolve_name(self, name):
        for cand in (name, alias(name), name.removeprefix("model.")):
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
                # computed on CPU and cast before the copy: a full fp32 copy of the 257152x2048 embedding on the GPU
                # (orig + new + cast, ~5 GB transient) ran densely packed servers out of memory
                if mode == "replace":
                    new = t.to(dtype=p.dtype)
                else:
                    orig = self.backup[key][1].float()
                    new = (orig + t.float() if mode == "add" else orig * t.float()).to(p.dtype)
                p.data.copy_(new)  # in place (same storage); host-to-device copy
                applied.append(key)
        return applied

    def restore(self, keys):
        ok = True
        h = hashlib.sha256()
        with torch.no_grad():
            for key in dict.fromkeys(keys):
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
    def set_compile(self, on):
        if on:
            if self.compiled_sample is None:
                self.compiled_sample = torch.compile(self.eager_sample, mode=self.args.compile_mode)
            self.policy._sample_actions = self.compiled_sample
        else:
            self.policy._sample_actions = self.eager_sample

    def infer(self, element, noise):
        """element = openpi LIBERO observation dict; noise [action_horizon, action_dim] float32.
        Returns executed actions [action_horizon, 7] (float64 numpy, after openpi output transforms)."""
        with torch.no_grad():
            out = self.policy.infer(element, noise=noise)
        return np.asarray(out["actions"])

    def make_element(self, obs, prompt):
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, RESIZE, RESIZE))
        wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, RESIZE, RESIZE))
        state = np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))
        return {"observation/image": img, "observation/wrist_image": wrist, "observation/state": state,
                "prompt": str(prompt)}

    # ---------------- episodes ----------------
    @staticmethod
    def _seed_all(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    @staticmethod
    def make_env(task, seed):
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=LIBERO_ENV_RESOLUTION,
                                 camera_widths=LIBERO_ENV_RESOLUTION)
        env.seed(seed)
        return env, task.language

    def run_episode(self, ep, *, recorder, out_dir, save_video, save_frames, tag, language):
        suite_name = ep["suite"]
        seed = int(ep.get("seed", 7))
        init_idx = int(ep.get("init_idx", 0))
        vid = int(ep["variant_id"]) if "variant_id" in ep else int(ep["task_idx"]) + 1
        task_idx = vid - 1
        meta = self.variants.meta.get((suite_name, vid), {})
        suite = self.variants.suite(suite_name)
        task = suite.get_task(task_idx)
        res = {"suite": suite_name, "variant_id": vid, "seed": seed, "init_idx": init_idx, "arm": tag,
               "name": task.name, "category": meta.get("category", ep.get("category")),
               "difficulty": meta.get("difficulty_level", ep.get("difficulty")),
               "success": False, "steps": 0, "failure": None}
        stem = f"{suite_name}_v{vid}_s{seed}_i{init_idx}_{tag}"
        t0 = time.time()
        env = None
        try:
            self._seed_all(seed)
            gen = torch.Generator(device="cpu").manual_seed(seed)  # flow-matching noise, fresh draw per query
            init_states = suite.get_task_init_states(task_idx)
            for attempt in range(5):
                try:
                    env, task_description = self.make_env(task, seed)
                    break
                except Exception as exc:
                    if exc.__class__.__name__ != "RandomizationError" or attempt == 4:
                        raise
            res["instruction_raw"] = task_description
            clean = getattr(env, "language_instruction", None)
            if isinstance(clean, (list, tuple)):
                clean = " ".join(clean)
            res["instruction_clean"] = clean
            mode = ep.get("language", language)
            if mode == "clean" and clean:
                task_description = str(clean).strip('"')
            elif mode == "strip":
                task_description = strip_variant_suffix(task_description)
            res["instruction"] = task_description
            max_steps = int(ep.get("max_steps", MAX_STEPS[suite_name]))
            frame_every = int(save_frames) if save_frames else 0
            fdir = None
            if frame_every and out_dir:
                fdir = Path(out_dir) / f"{stem}_frames"
                fdir.mkdir(parents=True, exist_ok=True)
            env.reset()
            obs = env.set_init_state(init_states[init_idx % len(init_states)])
            if recorder is not None:
                recorder.reset()
            frames, pending, done, t, n_q, trace, saved = [], [], False, 0, 0, [], []
            while t < max_steps + NUM_STEPS_WAIT:
                if t < NUM_STEPS_WAIT:
                    obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue
                st = t - NUM_STEPS_WAIT
                element = None
                if not pending or save_video:
                    element = self.make_element(obs, task_description)
                if not pending:
                    noise = torch.randn((self.action_horizon, self.action_dim), generator=gen, dtype=torch.float32).numpy()
                    chunk = self.infer(element, noise)
                    n_q += 1
                    pending = list(chunk[:REPLAN_STEPS])
                    if fdir is not None and st % frame_every == 0:
                        Image.fromarray(np.concatenate([element["observation/image"], element["observation/wrist_image"]], 1)).save(fdir / f"t{st:04d}.png")
                        torch.save({"element": element, "noise": noise, "seed": seed, "suite": suite_name,
                                    "variant_id": vid, "step": st, "arm": tag, "action_chunk": chunk,
                                    "replan_steps": REPLAN_STEPS}, fdir / f"t{st:04d}_inputs.pt")
                        saved.append(st)
                if save_video and st % 2 == 0:
                    frames.append(np.concatenate([element["observation/image"], element["observation/wrist_image"]], 1))
                a = pending.pop(0)
                trace.append(np.asarray(a, dtype=np.float64))
                obs, _, done, _ = env.step(np.asarray(a).tolist())
                if done:
                    break
                t += 1
            res["success"] = bool(done)
            res["steps"] = int(t - NUM_STEPS_WAIT + (1 if done else 0))
            res["n_queries"] = n_q
            res["termination"] = "success" if done else "max_steps"
            if not done:
                res["failure"] = "timeout"
            ah = np.asarray(trace, dtype=np.float32)
            res["action_sha1"] = hashlib.sha1(ah.tobytes()).hexdigest()[:16]
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
        compile_infer = bool(req.get("compile", False))  # default: eager (hooks work, same numerics as record)
        if record is not None and compile_infer:
            raise ValueError("record requires compile=false")
        self.set_compile(compile_infer)
        return out_dir, override, arms, record

    def run(self, req):
        out_dir, override, arms, record = self._common(req)
        recorder = Recorder(self, record) if record else None
        results, override_info = [], None
        try:
            for ep in req["episodes"]:
                for arm in arms:
                    with self.arm(arm, override, req.get("override_mode", "replace")) as info:
                        r = self.run_episode(ep, recorder=recorder, out_dir=out_dir,
                                             save_video=bool(req.get("save_video", False)),
                                             save_frames=req.get("save_frames", 0), tag=arm,
                                             language=req.get("language", "strip"))
                    if info is not None:
                        override_info = info
                        r["restore_verified"] = info["restore_verified"]
                    results.append(r)
        finally:
            if recorder is not None:
                recorder.close()
            self.set_compile(False)
        return {"ok": True, "results": results, "override": override_info,
                "gpu_max_reserved_gb": torch.cuda.max_memory_reserved() / 1e9}

    def probe(self, req):
        """inputs: *_inputs.pt files from save_frames. Per arm and input: action chunk [10,7] (executed space),
        optional activations (record spec) and attention maps (attention: {"layers": [expert layers]}, <=4x4)."""
        out_dir, override, arms, record = self._common(req)
        out_dir = Path(out_dir or (WSP / "runs/probe_tmp"))
        out_dir.mkdir(parents=True, exist_ok=True)
        attn = req.get("attention")
        if attn and (len(attn.get("layers", [])) > 4 or len(req["inputs"]) > 4):
            raise ValueError("attention maps limited to <=4 inputs and <=4 layers")
        recorder = Recorder(self, record) if record else None
        results, override_info = [], None
        try:
            for arm in arms:
                with self.arm(arm, override, req.get("override_mode", "replace")) as info:
                    for i, path in enumerate(req["inputs"]):
                        inp = torch.load(path, map_location="cpu", weights_only=False)
                        noise = inp["noise"]
                        if "seed_override" in req:
                            g = torch.Generator(device="cpu").manual_seed(int(req["seed_override"]))
                            noise = torch.randn(noise.shape, generator=g, dtype=torch.float32).numpy()
                        if recorder is not None:
                            recorder.reset()
                        cap = AttnCapture(self.model, attn["layers"]) if attn else None
                        with (cap if cap is not None else contextlib.nullcontext()):
                            action = self.infer(inp["element"], noise)
                        r = {"arm": arm, "input": path, "idx": i}
                        stem = out_dir / f"probe_{arm}_{i}"
                        np.save(f"{stem}_action.npy", action)
                        r["action"] = f"{stem}_action.npy"
                        r["matches_saved_chunk"] = bool(np.array_equal(action, inp.get("action_chunk")))
                        if recorder is not None:
                            r["activations"] = f"{stem}_acts.npz"
                            recorder.dump(r["activations"])
                        if cap is not None:
                            res_attn = cap.result()
                            for l, m in res_attn.items():
                                np.save(f"{stem}_attn_L{l}.npy", m)
                            r["attention"] = {int(l): f"{stem}_attn_L{l}.npy" for l in res_attn}
                            pm = self.last_prefix_mask
                            r["prefix_layout"] = {"base": [0, N_IMG_TOK], "wrist": [N_IMG_TOK, 2 * N_IMG_TOK],
                                                  "rwrist(masked)": [2 * N_IMG_TOK, 3 * N_IMG_TOK],
                                                  "lang": [3 * N_IMG_TOK, int(pm.numel())],
                                                  "n_lang_valid": int(pm[3 * N_IMG_TOK:].sum())}
                        results.append(r)
                if info is not None:
                    override_info = info
        finally:
            if recorder is not None:
                recorder.close()
            self.set_compile(False)
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
    ap.add_argument("--ckpt", default=str(WSP / "ckpt/pi05_libero_pytorch"))
    ap.add_argument("--compile-mode", default="default")
    ap.add_argument("--random-init", action="store_true", help="DEBUG: random weights (pipeline test only)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    PolicyServer(args).serve(args.sock)


if __name__ == "__main__":
    main()
