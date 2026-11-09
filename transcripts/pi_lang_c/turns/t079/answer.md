At half strength, the goal diagnostic set improves from 49 to 59 successes, with no regressions. I will now restore the exact confirmed full-strength file and test the same 80 pairs to complete the behavioral dose-response.
```python
install_recipe(F,"F_restored_full")
cur=np.load(W/"edited"/(E_NAME+".npy"),mmap_mode="r")
best=np.load(W/"best_F_dev"/(E_NAME+".npy"),mmap_mode="r")
same=all(np.array_equal(cur[i:i+8192],best[i:i+8192]) for i in range(0,len(cur),8192))
assert same
print("Restored exact confirmed file:",same)
print("Current tensor edits:",list(edited()))
print("Half-dose diagnostic:",paired_stats(pair_roll(33)))
with (W/"notes.md").open("a") as f:
    f.write("\n## Half-dose behavioral ablation (pass33)\n80 diagnostic goal cases:49->59, +12.5pp SE3.72, "
            "10 recoveries/0 regressions. F restored afterward and checked elementwise equal to best_F_dev/. "
            "Current active edit is FULL F, not half. Pass34 will test full F on exactly the same80 cases.\n")
```
