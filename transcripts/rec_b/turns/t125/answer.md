I will match the probe controls by instruction and gripper command where possible. Earlier apparent failure features were partly confounded by open-versus-closed commands, so the final audit should avoid repeating that comparison.
```python
soup_instruction = "pick up the alphabet soup and place it in the basket"
soup_diag = [
    r for r in r1 if r["model"] == "original" and r["instruction"] == soup_instruction
]
print("Exact-task diagnostic records:")
for r in soup_diag:
    print({k: r[k] for k in ("id", "grip", "seed", "success", "steps", "grip_flips")})

# Inspect only previously generated action traces and saved-input filenames.
audit_candidates = [(1, r) for r in soup_diag]
audit_candidates += [
    (47, r) for r in r47
    if r["model"] == "original" and r["id"] in {735, 744}
    and r["grip"] == "open" and r["seed"] == 7
]
for k, r in audit_candidates:
    ap = Path(r["actions_npy"])
    if not ap.is_relative_to(W):
        ap = W / str(ap).split("/work/", 1)[1]
    a = np.load(ap)
    d = W / f"roll_{k}_frames" / Path(r["frames_dir"]).name
    saved = sorted(d.glob("*_inputs.pt"))
    command_summary = []
    for p in saved:
        t = int(p.name.split("_")[0][1:])
        if t + 5 <= len(a):
            command_summary.append((t, round(float(a[t:t+5, 6].mean()), 3)))
    print("commands", r["id"], r["grip"], "shape", a.shape,
          "(step, mean_gripper_first5)", command_summary)

for dirname in ("probe_27", "probe_28"):
    p = W / dirname
    print(dirname, "exists", p.exists(),
          "files", [q.name for q in sorted(p.iterdir())[:8]] if p.exists() else [])
print("Prior probe request:",
      (W / "gripper_image_probe_request.json").read_text()[:1200])
```