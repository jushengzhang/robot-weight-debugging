The pooled paraphrase result is −3.1 points over 256 episodes, exceeding the allowed loss. This candidate will not be retained. I’ll complete the already planned third tagged set for a fixed-size efficacy estimate, then restore the original weights.
```python
slang=paired_summary([38,39])
print("Pooled language check:",json.dumps(slang,indent=2))
with (W/"notes.md").open("a") as f:
    f.write("\n### Pass39 frozen-gate language validation B; safety criterion failed\n"+
            json.dumps(paired_summary([39]))+"\n"
            "Pooled language A+B: "+json.dumps(slang)+"\n"
            "Language loss3.125pp exceeds approximately2pp allowed loss. Frozen gate "
            "will NOT be retained. Pass40 will complete preplanned rawC; then clear "
            "ALL edits and submit original weights. No other candidate had a "
            "validated significant tagged success gain.\n")
print("VALIDATION_C_EPISODES:")
print(json.dumps(json.loads((W/"validation_c_episodes.json").read_text()),
                 separators=(",",":")))
```
