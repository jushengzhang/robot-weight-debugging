This 24-episode sample shows only a one-episode clean-instruction gain, and all four camera variants fail in both conditions. That is too small and difficult a sample to estimate the benchmark gap reliably. I will inspect the paired inputs and activation files, then add a larger development sample while keeping the validation sets untouched.
```python
def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
r1 = read_jsonl(W/"roll_1.jsonl")
r2 = read_jsonl(W/"roll_2.jsonl")
print("RECORD EXAMPLE:", r1[0])
aidx1 = json.loads((W/"roll_1_acts"/"index.json").read_text())
print("ACT INDEX TYPE:", type(aidx1).__name__)
print(str(aidx1)[:1800])
raw_paths = sorted((W/"roll_1_frames").glob("*/t0000_inputs.pt"))
clean_paths = sorted((W/"roll_2_frames").glob("*/t0000_inputs.pt"))
print("INPUTS:", len(raw_paths),len(clean_paths))
inp = torch.load(raw_paths[0], map_location="cpu", weights_only=False)
print("INPUT KEYS:", {k:(tuple(v.shape),str(v.dtype)) if torch.is_tensor(v) else str(v)[:180] for k,v in inp.items()})
print("DEVELOPMENT EPISODES:", json.dumps(eps("development")))
with (W/"notes.md").open("a") as f:
    f.write("\nPass1-2: original raw12/24 vs strip13/24. Diagnostic set has camera0/4 in both; small/low-power gap. "
            "Saved first-observation pairs, attention episode means for both experts. Development split remains separate from validation.\n")
```
