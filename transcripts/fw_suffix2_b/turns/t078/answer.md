The prefix-only projection is worse on both tagged alignment and clean drift, so I will discard it and restore the broader rank-256 combined repair. Next I will test that candidate on 96 disjoint paraphrased instructions, where there is no suffix to ignore and the instruction must still be followed.
```python
install_hybrid_basis("rich_context_projection.pt",256,1.)
with (W/"notes.md").open("a") as f:
    f.write("\nPass27 prefix-only hybrid rejected: executed gap ratio.7898 vs broad hybrid.6543, normalized arm gap ratio.9443 vs.8996; clean/language drift.00877/.03072. Restore broad rank256 full-dose hybrid. Next development safety:96 previously unused Language Instructions variants from language_development.json. Candidate selection thereafter must rely on rollouts, not further diagnostic action-distance refinements. Reserved validation splits still unused.\n")
print("ACTIVE",len(edited()),json.loads((W/"current_candidate.json").read_text()))
```
