The earlier video nuisance directions came from episode averages, so robot trajectory differences may have contaminated them. Before using those directions for another edit, I’ll obtain residuals from matched initial observations with identical proprioception and instructions, plus the saved light controls. This will separate texture differences from differences caused by prior actions.
```python
pair_paths = json.loads((W / "initial_probe_paths.json").read_text())
light_paths = json.loads((W / "light_initial_paths.json").read_text())
print("Initial-pair paths:", len(pair_paths), "Light paths:", len(light_paths))
assert isinstance(pair_paths, list) and isinstance(light_paths, list)
assert len(pair_paths) % 2 == 0

# Short aliases point to untouched, forward-tool-generated policy inputs.
alias_dir = W / "i"
alias_dir.mkdir(exist_ok=True)
all_paths = pair_paths + light_paths
aliases = []
for i, src in enumerate(all_paths):
    src = Path(src)
    assert src.is_file() and str(src).startswith(str(W))
    dst = alias_dir / f"{i}.pt"
    if not dst.exists():
        dst.symlink_to(src)
    aliases.append(str(dst))

manifest = {
    "n_pair_inputs": len(pair_paths),
    "n_light_inputs": len(light_paths),
    "source_paths": all_paths,
    "aliases": aliases
}
(W / "fixed_residual_probe_manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps({
    "type":"probe", "model":"original", "inputs":aliases,
    "return":{"actions":True, "layers":list(range(30)), "what":"resid", "noise_seed":17}
}, separators=(",",":")))
```
