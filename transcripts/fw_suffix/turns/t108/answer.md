The first paraphrase set loses 3 successes out of 128 (−2.3 points, paired SE 1.7). I’ll record that result and run the second reserved paraphrase set so the safety assessment does not rest on one batch.
```python
s38=paired_summary([38])
print(json.dumps(s38,indent=2))
with (W/"notes.md").open("a") as f:
    f.write("\n### Pass38 frozen-gate language validation A\n"+
            json.dumps(s38)+"\n"
            "Paraphrase point estimate -2.34pp; no gate tuning or weight changes. "
            "Complete languageB and rawC before final retain/revert decision.\n")
print("LANGUAGE_B_EPISODES:")
print(json.dumps(json.loads((W/"language_b_episodes.json").read_text()),
                 separators=(",",":")))
```
