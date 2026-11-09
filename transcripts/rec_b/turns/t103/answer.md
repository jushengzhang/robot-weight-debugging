I will now verify the sham numerically and, if it is exact, increase only the same four visual-change directions from 1.5-times to 3-times gain. Keeping the subspace fixed makes this a dose test rather than a search for a new outcome-fitted direction.
```python
r40=json.loads((W/"probe_40"/"results.json").read_text())
b40={(r["idx"],r["arm"]):r for r in r40}
mx=0.0
for i in range(8):
    ro,re=b40[i,"orig"],b40[i,"edit"]
    ao,ae=np.load(local_path(ro["action"])),np.load(local_path(re["action"]))
    err=float(np.max(np.abs(ae-ao)))
    mx=max(mx,err)
    print("Sham input",i,"max_action_difference",err)
assert mx==0, "Sham is not exact; investigate before editing."
v2=set_content_edit(2.0)
print("V17 rank4 gain3x; relative whole-tensor change:",
      round(float((v2-v_orig).norm()/v_orig.norm()),4))
with (W/"notes.md").open("a") as f:
    f.write("\n## bf16 sham, pass40\n"
            "Identical V17 bf16 overlay gives exactly identical predicted actions on8 inputs. "
            "Repeated fragile outcomes are not caused by a bf16 loader mismatch. "
            "Candidate M now alpha2 (same rank4 basis, gain3x) for dose-response.\n")
```
