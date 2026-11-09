Layer 11 head 19 is the strongest visual-attention redistribution in both inspected pairs. Its peak visual-token mass rises from about 0.45 to 0.61 under failed paraphrases, while entropy falls in both cases. I will reduce only this action-query head’s logit scale to 75%, testing whether softer visual grounding reduces the instruction-induced misdirection without removing the head’s values or semantic features.
```python
def set_head_temperature(layer,head,scale,label=None):
    clear_edits()
    n=f"mot.mixtures.action.blocks.{layer}.self_attn.norm_q.weight"
    o=load_original(n)
    t=o.clone()
    t[128*head:128*(head+1)]*=scale
    save_edit(n,t)
    meta={"label":label or f"action_L{layer}_H{head}_query_scale{scale}",
          "name":n,"layer":layer,"head":head,"scale":scale,
          "changed_values":int((t!=o).sum()),
          "relative_delta":float((t-o).norm()/o.norm())}
    (W/"active_candidate.json").write_text(json.dumps(meta,indent=2))
    return meta
print(set_head_temperature(11,19,0.75))
with (W/"notes.md").open("a") as f:
    f.write("\n## Action visual-grounding candidate from fixed-input attention maps\n"
            "- At L11 (first sharp residual divergence), H19 has greatest mean visual distribution L1 difference across two fixed-observation pairs: .678. Both failed prompts concentrate attention: peak mass .448->.609 and .445->.606; entropy 1.802->1.195 and 1.755->1.535 nats.\n"
            "- Candidate: action L11 self_attn.norm_q.weight indices [2432:2560] scaled to .75. Only 128 query-scale entries change; no whole tensor scaling, no value/output deletion. Predict higher H19 attention entropy and reduced prompt-induced visual misdirection. Await fixed-input causal probe.\n")
```
