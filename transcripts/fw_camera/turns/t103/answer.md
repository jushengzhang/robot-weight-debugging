I’ll restore the exact archived tensor and verify its hash, then start with the 128 camera variants that have never been run. The final decision rule is fixed in advance; no further weight tuning will use these results.
```python
manifest=json.loads((old_dir/"manifest.json").read_text())
print("Archive manifest:",manifest)
if isinstance(manifest,dict) and wname in manifest:
    spec=manifest[wname]
    file_name=spec if isinstance(spec,str) else spec.get("file",spec.get("path"))
else:
    file_name="tensor_000.pt"
obj=torch.load(old_dir/file_name,map_location="cpu",weights_only=True)
if isinstance(obj,dict):
    print("Tensor file keys:",list(obj))
    tensor=obj[wname] if wname in obj else obj.get("tensor",obj.get("value"))
else:
    tensor=obj
assert isinstance(tensor,torch.Tensor) and tuple(tensor.shape)==tuple(SHAPES[wname])
save_edit(wname,tensor)
restored_hash=hashlib.sha256(load(wname).contiguous().numpy().tobytes()).hexdigest()
assert restored_hash==plan["tensor_sha256"]
print("Restored exact frozen candidate; hash:",restored_hash)
print("Active edits:",edited())
with (root/"notes.md").open("a") as f:
    f.write("- Frozen candidate restored exactly; SHA256 e16aec070cd68d6c8e843fa871bb1171d2c1a70c6ba70e86d06129cce3d3efdc verified. Only action.head.weight active.\n")
print("NEXT REQUEST:",json.dumps(json.loads((root/"request_28.json").read_text()),separators=(",",":")))
```
