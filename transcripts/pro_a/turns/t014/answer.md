Amplifying the instruction-reading head's output did not help overall: the paired change was 0.0 ± 2.5 points, with environment performance falling. I will discard that edit. A narrower hypothesis is that the head mixes too much visual content into its readout; I will sharpen only its query logits and verify whether attention shifts toward the instruction before testing behavior.
```python
r4=read_jsonl(W/"roll_4.jsonl")
def paired_records(rows):
    pairs={}
    for r in rows:
        key=(r["id"],r["seed"])
        arm=r["model"]
        pairs.setdefault(key,{})[arm]=r
    return pairs
def report_pairs(rows):
    pp=paired_records(rows)
    for typ in ["all","object","semantic","task","position","environment"]:
        sub=[a for a in pp.values() if "original" in a and "edited" in a
             and (typ=="all" or a["original"]["ptype"]==typ)]
        if not sub: continue
        dd=np.array([int(a["edited"]["success"])-int(a["original"]["success"]) for a in sub],float)
        print(typ,"n",len(dd),"diff/se_pp",round(100*dd.mean(),2),
              round(100*dd.std(ddof=1)/np.sqrt(len(dd)),2),
              "gain/loss",int((dd==1).sum()),int((dd==-1).sum()))
report_pairs(r4)
clear_edits()
q9n=f"{expert_root}.layers.9.self_attn.q_proj.weight"
q9=load_original(q9n).clone()
q9[4*256:5*256] *= 1.5
save_edit(q9n,q9)
snapshot("candidate_L9H4_q150")
print("Current edit:",edited())
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 4: candidate A, L9H4 output x1.5, dev128\n"
            "Original 69/128, edit69/128; paired 0.0 +/-2.5pp SE, 5 wins/5 losses. "
            "Task5->6, position3->4, environment17->13, object23->23, semantic21->23. "
            "Original-goal task failures13->12. REJECTED (no gain, environment damage).\n"
            "Candidate B: reset; L9H4 q_proj rows1024:1280 x1.5 only. "
            "Hypothesis: sharpen instruction-vs-visual selection, not output magnitude.\n")
```
