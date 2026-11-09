I will read the failed instructions and the saved-input format, then choose literal controls for the same tasks. I will also identify which words distinguish failures from successful paraphrases before proposing any weight edit.
```python
import re
def read_roll(k):
    return [json.loads(x) for x in (W / f"roll_{k}.jsonl").read_text().splitlines() if x.strip()]
r1 = read_roll(1)
print("Record keys:", list(r1[0]))
print("First record:", json.dumps(r1[0], ensure_ascii=True)[:2500])
print("Failed records, compact:")
for r in r1:
    if not r.get("success", False):
        print(json.dumps({k:v for k,v in r.items() if k in ["suite","id","variant_id","seed","instruction","success","failure_mode","task","name"]}, ensure_ascii=True))
idx1 = json.loads((W / "roll_1_acts/index.json").read_text())
print("Activation index type:", type(idx1).__name__)
print("Activation index example:", str(idx1)[:1200])
inputs1 = sorted((W / "roll_1_frames").rglob("*_inputs.pt"))
print("Saved inputs:", len(inputs1), str(inputs1[0].relative_to(W)))
d0 = torch.load(inputs1[0], map_location="cpu", weights_only=False)
print("Input type:", type(d0).__name__)
if isinstance(d0, dict):
    print("Input keys/shapes:", {k:(tuple(v.shape) if hasattr(v,"shape") else str(type(v))) for k,v in d0.items()})
def task_of(v):
    n = v["name"]
    return n.split("_language_")[0] if "_language_" in n else re.split(r"_view_|_initstate_|_table_|_light_|_noise_|_layout_", n)[0]
print("Goal language task counts:", collections.Counter(task_of(v) for v in lang if v["suite"] == "libero_goal"))
print("Low-perturbation literal goal controls:")
for task in sorted(set(task_of(v) for v in lang if v["suite"] == "libero_goal")):
    pool = [v for v in variants if v["suite"] == "libero_goal" and v["category"] != "Language Instructions" and v["difficulty"] == 1 and v["name"].startswith(task+"_")]
    print(task, [(v["id"],v["category"],v["name"][len(task):]) for v in pool[:2]])
```
