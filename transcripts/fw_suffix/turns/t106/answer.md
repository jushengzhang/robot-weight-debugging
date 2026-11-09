The pooled clean result is now −0.8 points over 256 episodes, with substantial uncertainty. I’ll record both tagged and clean estimates, then load the first reserved paraphrase set to test whether the gate preserves legitimate instruction variation.
```python
summaries={"raw_ab":paired_summary([34,35]),
           "strip_a":paired_summary([36]),
           "strip_b":paired_summary([37]),
           "strip_ab":paired_summary([36,37])}
print(json.dumps(summaries,indent=2))
(W/"validation_interim.json").write_text(json.dumps(summaries,indent=2))
with (W/"notes.md").open("a") as f:
    f.write("\n### Pass35 frozen-gate validation B (raw)\n"+
            json.dumps(paired_summary([35]))+"\n"
            "Raw A+B: "+json.dumps(summaries["raw_ab"])+"\n"
            "### Passes36-37 frozen-gate clean checks (strip A+B)\n"+
            json.dumps(summaries["strip_ab"])+"\n"
            "Candidate still frozen. Remaining: languageA/B, rawC. No significant "
            "tagged success improvement so far; final retention criteria unchanged.\n")
print("LANGUAGE_A_EPISODES:")
print(json.dumps(json.loads((W/"language_a_episodes.json").read_text()),
                 separators=(",",":")))
```
