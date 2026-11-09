The half-strength descriptive edit produces six rescues and no regressions on the exposure panel, a promising but development-only result. I will preserve that candidate, then test full strength on the identical panel to determine whether stronger semantic replacement helps or starts erasing useful distinctions.

```python
print("pass11 valid summary:",paired_summary(11))
for key,a,b in paired_records(11):
    if a["success"]!=b["success"]:
        print("rescue",key,a["instruction"])
shutil.copytree(W/"edited",W/"candidate_half_weights",dirs_exist_ok=True)
candidate_full_meta=apply_alias_groups([(direct_aliases,1.0),(descriptor_aliases,1.0)],"candidate_descriptor_full")
# Quantify first-action movement toward a successful same-task reference.
refs={}
for r in diag_records:
    if r["suite"]=="libero_goal" and r["success"] and r["difficulty"]<=2:
        refs.setdefault(task_of(r),r)
for key,a,b in paired_records(11):
    if a["success"]!=b["success"] and task_of(a) in refs:
        refa=np.load(refs[task_of(a)]["actions_npy"])[:5]
        aa=np.load(a["actions_npy"])[:5]
        ab=np.load(b["actions_npy"])[:5]
        print("initial-action reference RMS",a["id"],
              round(action_dist(aa,refa),4),"->",round(action_dist(ab,refa),4))
with (W/"notes.md").open("a") as f:
    f.write("\nPass11 D7 full + descriptive10 half:48 requested pairs,10 missing; "
            f"valid {paired_summary(11)}. All6 rescues in goal, no regressions. "
            "Saved candidate_half_weights. Testing descriptive full dose on same panel.\n")
```
