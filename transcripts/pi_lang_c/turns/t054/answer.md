The second batch shows a larger gain, +4.7 points, but also reports several execution errors. I will preserve the frozen recipe, check only the error metadata, and continue the predeclared validation.
```python
print("Pass21 runtime error metadata")
for r in read_roll(21):
    if r.get("termination") not in ("success","max_steps"):
        print({k:v for k,v in r.items() if k not in ("instruction","name","actions_npy","frames_dir","activations")})
print("CONFIRM4",json.dumps(json.loads((W/"confirm_4_episodes.json").read_text()),separators=(",",":")))
with (W/"notes.md").open("a") as f:
    f.write("\nF confirmation batch2/pass21: reported 115->121 /128, 8 recoveries/2 regressions, "
            "+4.69 pp SE2.45. Several execution errors need classification; retain reported all-pair score "
            "as primary until errors understood. F unchanged; continue predeclared validation.\n")
```
