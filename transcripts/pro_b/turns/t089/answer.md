I will use reversible swaps of object and spatial-relation token embeddings as a diagnostic intervention. The images and noise remain fixed, so any visual-residual difference is caused by instruction content. These swaps will be removed before testing a repair; they are not a proposed final edit.
```python
pairs=[("black","white"),("red","yellow"),("bowl","mug"),("left","right"),
       ("front","back"),("top","bottom"),("tomato","salad"),("sauce","dressing"),
       ("wine","milk"),("basket","tray"),("open","close")]
token_pairs=[]
for a,b in pairs:
    for prefix in [""," "]:
        ia=sp.encode(prefix+a,out_type=int)
        ib=sp.encode(prefix+b,out_type=int)
        if len(ia)==len(ib)==1 and ia[0]!=ib[0]:
            token_pairs.append((ia[0],ib[0],prefix+a,prefix+b))
# Only disjoint pairs.
seen=set(); distinct=[]
for a,b,wa,wb in token_pairs:
    if a in seen or b in seen:continue
    seen.update([a,b]);distinct.append((a,b,wa,wb))
en=LP+"embed_tokens.weight"
E=load_original(en)
Es=E.clone()
for a,b,_,_ in distinct:
    Es[a]=E[b];Es[b]=E[a]
save_edit(en,Es)
print("Swaps",[(a,b,wa,wb) for a,b,wa,wb in distinct])
print("Path alias available",Path("/work").exists(),"WORK",W)
# 64 observations covering all 40 task identities.
lang_inputs=[x for x in selected if x["ptype"] in ["task","semantic"]]
assert len(lang_inputs)==64
for x in lang_inputs:
    # Saved inputs, not modified inputs.
    relative=Path(x["input"]).relative_to(W)
    x["short_input"]="/work/"+str(relative) if Path("/work").exists() else x["input"]
(W/"controlled_language_inputs64.json").write_text(json.dumps(lang_inputs,indent=2))
torch.save({"pairs":distinct},W/"subspaces"/"controlled_token_swaps.pt")
print("Probe request:")
print(json.dumps({"type":"probe","model":"both","inputs":[x["short_input"] for x in lang_inputs],
                  "return":{"layers":[12,13,14,15,16,17],
                            "vlm_layers":[9,11,12,13,14,15,16,17],"what":"resid"}}))
del Es,E
with (W/"notes.md").open("a") as f:
    f.write("\n## Controlled multi-scene language diagnosis\n"
            "Temporary disjoint token-embedding swaps of objects, colors and spatial relations on64 saved initial observations, "
            "covering40 task identities. Images/noise unchanged. Probe late VLM and expert residuals to isolate instruction-caused "
            "visual directions without the earlier task-vs-semantic image confound. Swaps are diagnostic only.\n")
```
