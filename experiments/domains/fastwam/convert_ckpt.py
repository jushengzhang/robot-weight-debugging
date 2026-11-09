"""Convert the FastWAM .pt checkpoint ({'mot': state_dict, 'proprio_encoder': state_dict, ...}) to one flat
safetensors file with keys '<top>.<param>' (lazy per-tensor loading for the editor's kernel)."""
import json, sys, torch
from safetensors.torch import save_file

src, dst = sys.argv[1], sys.argv[2]
obj = torch.load(src, map_location="cpu", mmap=True, weights_only=False)
sd = {}
def walk(o, pre):
    if torch.is_tensor(o):
        sd[pre] = o.contiguous()
    elif isinstance(o, dict):
        for k, v in o.items():
            walk(v, f"{pre}.{k}" if pre else str(k))
walk(obj, "")
print(len(sd), "tensors;", round(sum(v.numel() for v in sd.values()) / 1e9, 3), "B params; top-level", list(obj))
pref = {}
for k, v in sd.items():
    p = ".".join(k.split(".")[:3])
    pref[p] = pref.get(p, 0) + v.numel()
print(json.dumps({k: round(n / 1e6, 1) for k, n in sorted(pref.items(), key=lambda x: -x[1])[:30]}))
print("dtypes", sorted({str(v.dtype) for v in sd.values()}))
print("example keys", list(sd)[:5], list(sd)[-5:])
assert len(sd) > 100
save_file(sd, dst)
print("saved", dst)
