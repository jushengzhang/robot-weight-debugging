The first independent batch contradicts the development result: the edit lost six successes, or 4.7 points. I’ll record that negative result and continue the preplanned validation without changing weights; one batch is not enough to estimate the overall effect reliably.
```python
s36=paired_rollout(36)
with (W/"notes.md").open("a") as f:
    f.write("\n### Pass36 frozen early-video gate, tagged validation0\n"+json.dumps(s36)+
            "\nNegative independent result. Candidate remains frozen; continue preplanned validation, "
            "do not include development outcomes in final gain estimate.\n")
used_development={(x["suite"],x["id"]) for k in ["diagnostic","language_diag","development","language_dev"] for x in splits[k]}
reserved={(x["suite"],x["id"]) for k in plan["raw_splits"]+plan["language_splits"] for x in splits[k]}
assert used_development.isdisjoint(reserved)
print("Independent validation:",s36)
print("Validation/development variant overlap:",len(used_development & reserved))
print("NEXT:",json.dumps([{k:x[k] for k in ("suite","id","seed")} for x in splits["tag_validation_1"]],separators=(",",":")))
```
