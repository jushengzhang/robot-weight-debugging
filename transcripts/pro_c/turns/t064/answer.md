The expert MLP edit gained four successes without losses in semantic or task totals, but it did not reduce old-task completions and is not statistically conclusive. I will test full removal of the same four directions. This is a dose test, not a new subspace selected from the rollout outcomes.
```python
import shutil
backup=W/"candidate_H_half"
if backup.exists():shutil.rmtree(backup)
shutil.copytree(W/"edited",backup)
for l in [12,13]:
    n=f"{EX}.layers.{l}.mlp.down_proj.weight"
    v=load_original(n);u=expert_scene_time[l]
    save_edit(n,v-u@(u.T@v))
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass21: H half dose\n71->75/128, +3.1pp SE2.2; env14->16, object18->19, position4->5, semantic27->27, task8->8. Old-goal failures11 unchanged, loop failures30 unchanged. Inconclusive but no retention loss on this development sample. Saved candidate_H_half; testing full dose same basis.\n")
print("H full dose relative changes",[(n.split(".layers.")[1].split(".")[0],round(float((load(n)-load_original(n)).norm()/load_original(n).norm()),4)) for n in edited()])
```
