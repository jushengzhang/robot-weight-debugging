I will compare the two storage formats on all 16 saved inputs, confirm that untargeted control prompts remain unchanged, and inspect the citrus pair’s instructions. Then I will retry the 25 validation pairs that failed to execute, using their original variants and seeds.
```python
rr23,pi23=probe_index(23)
fmt_eq=[]
for i in range(16):
    x,y=action_at(pi22,"edit",i),action_at(pi23,"edit",i)
    fmt_eq.append(np.array_equal(x,y))
print("Compact vs exact edited actions:",Counter(fmt_eq),
      "maxabs",max(float(np.max(np.abs(action_at(pi22,"edit",i)-action_at(pi23,"edit",i)))) for i in range(16)))
dev_lookup={(r["suite"],r["id"]):r for k in (1,2,4) for r in read_roll(k)}
untouched=[]
for i,path in enumerate(probe_inputs):
    m=re.search(r"(libero_[a-z0-9]+)_v(\d+)_s",Path(path).parent.name)
    r=dev_lookup[(m.group(1),int(m.group(2)))]
    hit=set(sp.encode(r["instruction"]+"\n",add_bos=True)) & set(spec)
    if not hit:
        untouched.append(i)
    if i in (14,15):
        print("Citrus comparison",i,"success",r["success"],"instruction",repr(r["instruction"]),
              "target_rows",sorted(hit))
print("Untargeted prompt indices:",untouched)
print("Original vs exact edit on untargeted controls:",
      Counter(np.array_equal(action_at(pi23,"orig",i),action_at(pi23,"edit",i)) for i in untouched))
# Verify replay conditions without executing the network.
obs14=torch.load(probe_inputs[14],map_location="cpu",weights_only=False)
print("Saved policy-input fields:",list(obs14) if isinstance(obs14,dict) else type(obs14).__name__)
def schema(v):
    if isinstance(v,dict): return {k:schema(x) for k,x in v.items()}
    if torch.is_tensor(v) or isinstance(v,np.ndarray): return (type(v).__name__,tuple(v.shape))
    return type(v).__name__
print("Saved input schema:",str(schema(obs14))[:1300])
with (W/"notes.md").open("a") as f:
    f.write(f"\nProbe23 exact-source vs probe22 compact: {sum(fmt_eq)}/16 edited action chunks bit-identical. "
            f"Untargeted controls {untouched} also original/edit bit-identical. "
            "Current artifact remains exact-source fp32; no incidental changes outside the 20 rows.\n")
print("\nERROR RETRY REQUEST:")
print(json.dumps(json.loads((W/"validation_error_retry.json").read_text()),separators=(",",":")))
```
