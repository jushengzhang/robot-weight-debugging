"""Export the editable pi0.5 weights exactly as held by the policy server (after openpi's
to_bfloat16_for_selected_params) to ONE flat safetensors file keyed by the server's parameter names, plus an
index (name -> shape, dtype, numel, group). Run with the wsp_pi05 venv (CPU only, ~1 min):
    venv/bin/python export_weights.py ckpt/pi05_libero_pytorch weights/pi05_libero_flat.safetensors
"""
import collections
import dataclasses
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
W = Path(__file__).resolve().parent
os.environ.setdefault("OPENPI_DATA_HOME", str(W / "cache/openpi"))
import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from openpi.training import config as _config  # noqa: E402

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
cfg = _config.get_config("pi05_libero")
cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None))
model = cfg.model.load_pytorch(cfg, str(src / "model.safetensors"))
model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")  # same as policy_config
sd = {n: p.detach().contiguous() for n, p in model.named_parameters()}  # deduplicated (tied lm_head/embed once)
aliases = {n: m for n, m in ((n, n) for n, _ in model.named_parameters(remove_duplicate=False)) if n not in sd}

def group(n):
    for pre, g in (("paligemma_with_expert.paligemma.model.language_model.", "vlm"),
                   ("paligemma_with_expert.gemma_expert.model.", "expert"),
                   ("paligemma_with_expert.gemma_expert.lm_head", "expert_lm_head(unused)"),
                   ("paligemma_with_expert.paligemma.model.vision_tower.", "vision"),
                   ("paligemma_with_expert.paligemma.model.multi_modal_projector.", "projector"),
                   ("paligemma_with_expert.paligemma.lm_head", "vlm_lm_head(unused)")):
        if n.startswith(pre):
            return g
    return "action_heads"

idx = {n: {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", ""), "numel": t.numel(), "group": group(n)}
       for n, t in sd.items()}
g = collections.Counter()
for v in idx.values():
    g[v["group"]] += v["numel"]
dst.parent.mkdir(parents=True, exist_ok=True)
save_file(sd, str(dst))
meta = {"source": str(src), "n_tensors": len(sd), "total_params": sum(g.values()), "params_per_group": dict(g),
        "dtypes": dict(collections.Counter(v["dtype"] for v in idx.values())),
        "tied_aliases_not_stored": list(aliases)}
(dst.parent / "index.json").write_text(json.dumps({"meta": meta, "tensors": idx}, indent=1))
print(json.dumps(meta, indent=1))
