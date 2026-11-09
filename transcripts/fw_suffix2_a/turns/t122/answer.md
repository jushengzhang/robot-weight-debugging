The active edit is the planned one-tensor video projection. Pass 39 showed no overall change on the development safety set, but paraphrase success fell by 2 of 48 episodes, so safety remains uncertain. I will verify the frozen weights and record the result before testing fresh tagged variants.
```python
print("CANDIDATE HELPER:")
print((W / "candidate_helpers.py").read_text())
print("ANALYSIS HELPER:")
print((W / "analysis_helpers.py").read_text())
print("FROZEN MANIFEST:")
print((W / "frozen_video_orthogonal_rank6_manifest.json").read_text())

name = "mot.mixtures.video.text_embedding.0.weight"
frozen_dir = W / "frozen_video_orthogonal_rank6"
print("FROZEN FILES:", [p.name for p in frozen_dir.iterdir()])
current = load(name)
original = load_original(name)
snapshot = np.load(frozen_dir / (name + ".npy"))
print("Snapshot identical:", np.array_equal(current.numpy(), snapshot))
print("Relative weight change:",
      float(torch.linalg.vector_norm(current - original) /
            torch.linalg.vector_norm(original)))
del current, original, snapshot

rows39 = [json.loads(s) for s in (W / "roll_39.jsonl").read_text().splitlines() if s.strip()]
print("PASS39 record count:", len(rows39))
print("PASS39 first record:", json.dumps(rows39[0])[:1700])
print("REMAINING SPLITS:", {
    k: len(v) for k, v in splits.items()
    if k.startswith("language_validation") or k.startswith("tag_validation")
})

note = """
### Pass39 rank6 projection development safety
128 paired strip episodes including48 language:87->87 successes;6 gains/6 losses,
difference0.0pp,SE2.7pp. Clean(non-Language)59->61/80(+2.5pp);
Language30->28/48(-4.17pp), uncertain with this small sample.
Frozen one-tensor candidate unchanged. Independent validation remains:
40-43 tag_validation_2..5 raw;44-46 tag_validation_2..4 strip;
47-50 language_validation_0..3 raw. No further candidate tuning.
"""
with (W / "notes.md").open("a") as f:
    f.write(note)
```
