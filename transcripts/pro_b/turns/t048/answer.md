Layer 12 head 2 concentrates on the requested object (“tomato” receives 27% of its attention in one failed case), while layer 15 head 0 reads broader instruction content. I will double only these two heads’ output-projection blocks. This leaves their attention routing unchanged and tests whether strengthening the instruction information they already retrieve can overcome the scene’s habitual action.
```python
import shutil
head_gain = {12:[2], 15:[0]}
for l, hs in head_gain.items():
    n=f"{ex}.layers.{l}.self_attn.o_proj.weight"
    a=load_original(n)
    for h in hs:
        a[:,h*256:(h+1)*256] *= 2.0
    save_edit(n,a)
dst=W/"candidate_instruction_heads_g2"
dst.mkdir(exist_ok=True)
for p in (W/"edited").iterdir():
    if p.is_file(): shutil.copy2(p,dst/p.name)
lex_ids=json.loads((W/"lex_test_ids.json").read_text())
print("Head gain:", head_gain, "factor",2.0)
print("Test IDs:", lex_ids)
print("Test composition:", {p:sum(byid[i]["ptype"]==p for i in lex_ids)
                            for p in sorted({byid[i]["ptype"] for i in lex_ids})})
with (W/"notes.md").open("a") as f:
    f.write("\n## Candidate instruction-reading head gain\n"
            "L12H2 directly attends requested nouns (tomato0.273; wine/bottle route also strong); "
            "L15H0 has highest mean non-BOS instruction mass0.210. "
            "Double only these two expert o_proj column blocks. Other heads unchanged. "
            "Test on the existing diagnostic 68task+60controls set; prediction is fewer "
            "old-goal completions and more correct target selection.\n")
(W/"current_candidate.json").write_text(json.dumps({"name":"instruction_heads_g2","heads":head_gain,"gain":2.0}))
```
