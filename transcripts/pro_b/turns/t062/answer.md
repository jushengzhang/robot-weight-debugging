The nominally matched observations have different images, so their action differences are not a clean test of instruction use. I will create a controlled diagnostic through weights alone: temporarily swap the “tomato sauce” and “salad dressing” embedding rows, then probe the exact same saved images and noise. This tests whether changing object identity in language actually reaches the action expert.
```python
# A temporary semantic-swap diagnostic; these edits will not be retained as a repair.
emb=load_original(embname)
token_pairs=[(" tomato"," salad"),(" sauce"," dressing")]
swap_ids=[]
for a,b in token_pairs:
    ia,ib=sp.encode(a),sp.encode(b)
    print(repr(a),ia,repr(b),ib)
    assert len(ia)==len(ib)==1
    ia,ib=ia[0],ib[0]
    row=emb[ia].clone()
    emb[ia]=emb[ib]
    emb[ib]=row
    swap_ids.append((ia,ib))
save_edit(embname,emb)
del emb
# Use the saved initial and later observations in the two salad-scene episodes.
swap_inputs=[]
for x in pr[2][:2]:
    p=Path(x["input"])
    files=sorted(p.parent.glob("*_inputs.pt"))
    swap_inputs.extend(str(p) for p in files[:5])
print("Diagnostic inputs:",swap_inputs)
(W/"semantic_swap_inputs.json").write_text(json.dumps(swap_inputs))
with (W/"notes.md").open("a") as f:
    f.write("\n## Confound found; controlled language diagnosis planned\n"
            "Task/semantic observations at same (suite,task,init_idx) have identical "
            "robot state and noise but different images (salad example RGB RMS11/18). "
            "Prior paired activation contrasts are not pure instruction contrasts. "
            "Temporary swaps of tomato<->salad and sauce<->dressing embedding rows "
            "will test causal language use on identical saved images/noise; not a final edit.\n")
```
