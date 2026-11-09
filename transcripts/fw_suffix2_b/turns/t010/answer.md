I’ll build a replay set containing 24 tagged/clean pairs and 24 paraphrases, all using saved, unmodified inputs. I’ll also recover the prior detector’s direction and identify the proprioception subspace, which any text-only intervention should preserve.```python
import shutil
R = W / "replay"
R.mkdir(exist_ok=True)
language_paths = []
for v in splits["language_diag"]:
    sub = f"{v['suite']}_v{v['id']}_s{v['seed']}_original"
    language_paths.append(str(W / "roll_3_frames" / sub / "t0000_inputs.pt"))
replay_files = []
for prefix, paths in [("r", raw_paths), ("c", clean_paths), ("l", language_paths)]:
    for i, src in enumerate(paths):
        dest = R / f"{prefix}{i:02d}.pt"
        if not dest.exists():
            dest.symlink_to(src)
        replay_files.append(str(dest))
for a,b in zip(raw_paths, clean_paths):
    x,y = [torch.load(p, weights_only=False, map_location="cpu") for p in [a,b]]
    assert torch.equal(x["input_image"], y["input_image"])
    assert torch.equal(x["proprio"], y["proprio"])
    assert x["seed"] == y["seed"]
print("All24 matched image/proprio/seed checks passed.")
print("Replay paths:", json.dumps(["WORK/replay/" + Path(p).name for p in replay_files]))
prop = torch.cat([load_original("proprio_encoder.weight"),
                  load_original("proprio_encoder.bias")[:,None]], dim=1)
Qprop = torch.linalg.qr(prop.double(), mode="reduced").Q.float()
detectors = {}
for expert in ["action", "video"]:
    p = f"mot.mixtures.{expert}.text_embedding."
    a = torch.from_numpy(np.load(pc / (p+"0.weight.npy")))
    b = torch.from_numpy(np.load(pc / (p+"0.bias.npy")))
    changed_row = torch.where((a - load_original(p+"0.weight")).abs().sum(1) > 0)[0]
    i = int(changed_row[0])
    detectors[expert] = a[i] / a[i].norm()
    print(expert, "detector hidden", i, "gain", float(a[i].norm()), "bias", float(b[i]),
          "proprio overlap", float((Qprop.T @ detectors[expert]).norm()))
print("Detector cosine", float(detectors["action"] @ detectors["video"]))
print("Proprio span dimension:", Qprop.shape[1])
(W / "replay_paths.json").write_text(json.dumps(replay_files))
with (W / "notes.md").open("a") as f:
    f.write("\nPass3: diagnostic paraphrases19/24 (79.2%). Replay directory contains r00:r23, "
            "c00:c23, l00:l23. All initial raw/clean image, proprio and seeds exactly match.\n")
```
