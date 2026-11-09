I will summarize the oblique-projection probe and then prepare an untouched validation set. I will exclude every language variant already used in diagnosis or edit selection, and keep later confirmation variants separate so that the final decision is not based on the discovery sample.
```python
summary16=probe_compare(16)
(W/"probe16_comparison.json").write_text(json.dumps(summary16,indent=2))
parts=json.loads((W/"practice_partitions.json").read_text())
from collections import Counter
print("\nPractice partitions by category:")
for part,rows in parts.items():
    print(part,dict(Counter(r["category"] for r in rows)))
seen_variant={(r["suite"],int(r["id"])) for r in r1+r2+r3}
unseen_lang={part:[r for r in rows if r["category"]=="Language Instructions"
                   and (r["suite"],int(r["id"])) not in seen_variant]
             for part,rows in parts.items()}
print("Unseen language variants:",{k:len(v) for k,v in unseen_lang.items()})
print("All unmodified vs projected action-gap mean:",
      np.round(np.mean(summary16["action_metrics"],axis=0),6).tolist())
clear_edits()
print("Reverted geometric edit; active:",edited())
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 16\n"
            "- Tested oblique protected L11 projector at alpha .5 on all ten original instruction pairs; results saved in probe16_comparison.json. Reverted pending independent assessment; do not treat suppression of a diagnostic activation as proof of semantic repair.\n"
            "- Discovery limit reached: next stage will validate the minimal L11 H19 query-scale .75 candidate on variants excluded from all original diagnostic rollouts. It is only provisional (+.8 pp discovery, below SE), and will be rejected unless independent gain is clear and other categories are safe.\n")
```
