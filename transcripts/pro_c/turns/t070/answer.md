The illumination edit again gained three successes, but only by one standard error, and the gain was not concentrated in environment episodes. I will keep it as an unconfirmed lead and screen the half-dose expert MLP edit next.
```python
for n in list(edited()):drop_edit(n)
for l in [12,13]:
    n=f"{EX}.layers.{l}.mlp.down_proj.weight"
    v=load_original(n);u=expert_scene_time[l]
    save_edit(n,v-0.5*u@(u.T@v))
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass24: D0.5 fresh screen seed31023\n77->80/128,+2.3pp SE2.3. Env19->19/33, object29->30/31, semantic19->20/20, position5->6/22, task5->5/22. Old-goal10->11, loops33->29. Two samples show +3 each but neither significant, and fresh gain not concentrated in environment. Still only unconfirmed lead. Current H0.5 restored.\n")
print("H0.5 restored",edited())
```
