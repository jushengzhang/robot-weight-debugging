The original model succeeds on every sampled difficulty-1-to-3 goal instruction but only 13 of 31 difficulty-5 instructions. I will now inspect the failed wording, reported behaviors, and activation-file structure, then choose matched easy/hard examples for attention probes.

```python
def read_roll(k):
    return [json.loads(s) for s in (W / f"roll_{k}.jsonl").read_text().splitlines() if s.strip()]
r1 = read_roll(1)
print("record keys:", r1[0].keys())
print("first record:", json.dumps(r1[0], ensure_ascii=True)[:3500])
print("activation index:")
idx1 = json.loads((W / "roll_1_acts" / "index.json").read_text())
print(str(idx1)[:2500])
print("frame filenames:", [p.name for p in (W/"roll_1_frames").iterdir()][:5])
for r in r1:
    if not r.get("success", r.get("original", {}).get("success", True)):
        print(json.dumps(r, ensure_ascii=True)[:1500])
with (W / "notes.md").open("a") as f:
    f.write("\nPass 1: original, 80 goal language episodes (sample seed 17). "
            "60/80 success. Difficulties 1-3: 34/34; 4:13/15; 5:13/31. "
            "Recorded VLM/expert residuals at layers 0,3,6,9,12,15,17 and frames.\n")
```
