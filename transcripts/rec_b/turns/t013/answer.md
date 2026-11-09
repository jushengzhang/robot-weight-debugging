The broader maps suggest a visual-selection problem, but not a universal “base camera off” failure: layer 7 head 7 sometimes remains active during a loop. I will test the narrower hypothesis that strengthening this head’s selectivity helps ambiguous observations. The first intervention changes only its 256 query rows, leaving the other seven heads and all action scaling untouched.
```python
def reset_edits():
    for n in list(edited()): drop_edit(n)
def edit_q_heads(spec):
    # spec: {layer: {head: multiplier}}, always relative to the original checkpoint.
    for L,hm in spec.items():
        n=f"{EX}.layers.{L}.self_attn.q_proj.weight"
        t=load_original(n).clone()
        for h,a in hm.items(): t[h*256:(h+1)*256] *= a
        save_edit(n,t)
def edit_o_heads(spec):
    for L,hm in spec.items():
        n=f"{EX}.layers.{L}.self_attn.o_proj.weight"
        t=load_original(n).clone()
        for h,a in hm.items(): t[:,h*256:(h+1)*256] *= a
        save_edit(n,t)
tune_ids=[304,487,1158,1274,1102,935,148,620,992,1277,405,1259,
          322,647,815,1340,199,1300,11,27,109,140,65,126]
tune_eps=episodes_for(tune_ids)
print("tuning baseline",sum(R[s,g]["success"] for s in tune_ids for g in ["open","close"]),"/",len(tune_eps))
reset_edits()
edit_q_heads({7:{7:1.3}})
print("edited",list(edited()))
print("edited relative Frobenius norm",
      float((load(f"{EX}.layers.7.self_attn.q_proj.weight")-
            load_original(f"{EX}.layers.7.self_attn.q_proj.weight")).norm()/
             load_original(f"{EX}.layers.7.self_attn.q_proj.weight").norm()))
print("tune episodes:",json.dumps(tune_eps,separators=(",",":")))
(W/"tune_episodes.json").write_text(json.dumps(tune_eps))
with (W/"notes.md").open("a") as f:
    f.write("\n## Attention diagnosis, passes 2-4\nMatched L3 step50 maps: expert L7h7 base mass "
            "48/43% in failures vs93/79% in successes; more self attention in failures. "
            "At t0 and in other late loops the difference is not universal (one loop has84% base). "
            "L5h3 consistently wrist-dominated (~94-96% in late loops). L1 instruction sink normal in both. "
            "Working hypothesis: expert visual selection / reliance on local gripper view, not missing mask or "
            "simple gripper readout bias. First causal test Q L7h7 x1.3 (head only).\n")
```
