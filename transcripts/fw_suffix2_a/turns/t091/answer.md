Lowering the threshold did not help: the edited model lost two successes on the development set. I’ll restore the standard threshold and check clean instructions and paraphrases before spending more passes on it. The safety check will use the reserved language-development set plus stripped instructions from the tagged development set.
```python
s26 = paired_rollout(26)
restore_dense_candidate(experts=("video",), dose=1., threshold=.20)
safety_dev = splits["development"][:80] + splits["language_dev"]
safety_eps = [{k:x[k] for k in ("suite","id","seed")} for x in safety_dev]
(W / "safety_development_episodes.json").write_text(json.dumps(safety_eps))
with (W / "notes.md").open("a") as f:
    f.write("\n### Pass26 lower-threshold video gate rejected\n"+json.dumps(s26)+
            "\nRestored video-only threshold.20 standard dose. "
            "Next: paired strip safety on80 tagged development+48 language development variants.\n")
print("SAFETY EPISODES:", json.dumps(safety_eps, separators=(",",":")))
```
