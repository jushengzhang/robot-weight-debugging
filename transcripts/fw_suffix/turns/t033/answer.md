The outlier-neuron edit is too weak to establish a repair. I will instead measure the text-context nuisance directions directly using temporary weight-only readouts through the forward tool. Before that, I will collect a separate set of paraphrased instructions so the eventual projection can explicitly protect both ordinary clean language and paraphrases.
```python
clear_edits()
all_reserved={(v["suite"],v["id"]) for vs in splits.values() for v in vs}
lang_pool=[v for v in variants if v["category"]=="Language Instructions"
           and (v["suite"],v["id"]) not in all_reserved]
splits["language_diagnostic"]=balanced_set(lang_pool,32,"language-diagnostic")
(work/"splits.json").write_text(json.dumps(splits,indent=2))
lang_diag_episodes=[{"suite":v["suite"],"id":v["id"],"seed":2901+i}
                    for i,v in enumerate(splits["language_diagnostic"])]
print("Language diagnostic episodes:",json.dumps(lang_diag_episodes))
with (work/"notes.md").open("a") as f:
    f.write("\n## Next diagnostic: direct text-context mean readout\n"
            "Previous candidates did not materially restore clean actions; no repair selected. "
            "Plan temporary weight-only instrumentation in action block 0: uniform-query attention "
            "reads selected context coordinates, with other writes to those residual coordinates "
            "disabled. GELU(x)-GELU(-x)=x permits an exact linear readout through the existing "
            "text MLP, without editing frozen UMT5 or running any network locally. "
            "All instrumentation will be removed before repair evaluation. "
            "Collect 32 disjoint language-diagnostic paraphrases first; validation language sets remain unused.\n")
```
