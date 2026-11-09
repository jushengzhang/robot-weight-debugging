The late grounding heads are not interchangeable: layer 14 head 0 often puts about 40% of its attention on positional-register patches, whereas head 1 tracks ordinary image patches and also responds to noun changes. I will test whether the register-heavy route overrules local evidence by halving only head 0’s output, leaving the other grounding heads intact. This is a causal test, not a conclusion that register tokens are inherently harmful.
```python
EP="paligemma_with_expert.gemma_expert.model."
LP="paligemma_with_expert.paligemma.model.language_model."
VP="paligemma_with_expert.paligemma.model.vision_tower.vision_model."
pr=json.loads((Path(BENCH)/"practice_episodes.json").read_text())
if isinstance(pr,dict):
    print("Practice keys",list(pr)[:10])
    pr=pr.get("episodes",pr)
print("Practice count",len(pr),"example",str(pr[0])[:900])
manifest=json.loads((W/"split_manifest.json").read_text())
print("Split sizes",{k:len(v) for k,v in manifest.items()})
# Stratification uses only metadata, not outcomes.
diag=set(manifest["diagnostic"])
rng=np.random.default_rng(2619)
chosen=[]
for pt, count in [("task",32),("position",32),("environment",32),("object",16),("semantic",16)]:
    pool=[r for r in pr if r["id"] in diag and r["ptype"]==pt]
    for suite in sorted(set(r["suite"] for r in pool)):
        ss=[r for r in pool if r["suite"]==suite]
        order=rng.permutation(len(ss))
        chosen += [ss[int(j)]["id"] for j in order[:count//4]]
assert len(chosen)==128
(W/"register_head_test_ids.json").write_text(json.dumps(chosen))
name=EP+"layers.14.self_attn.o_proj.weight"
A=load_original(name)
Ae=A.clone()
Ae[:,0:256]*=0.5
save_edit(name,Ae)
bdir=W/"candidate_l14_register_head_half"
bdir.mkdir(exist_ok=True)
torch.save(Ae,bdir/(name+".pt"))
(W/"current_candidate.json").write_text(json.dumps({"name":"l14_register_head_half","heads":[[14,0]],"gain":0.5}))
with (W/"notes.md").open("a") as f:
    f.write("\n## Candidate late register-head attenuation\n"
            "Pure noun-swap probes show L14H0 is instruction-sensitive but allocates about40% attention to the five "
            "high-norm positional rows on three observations; H1 instead attends ordinary patches. "
            "Prior gain changed H0,H1,H3 together, so did not isolate this route. "
            "Test only L14H0 o_proj block x0.5 on a metadata-stratified diagnostic128: "
            "32task+32position+32environment+16object+16semantic. No other edit.\n")
print("Edit relative Frobenius norm",float((Ae-A).norm()/A.norm()))
print("Diagnostic rollout JSON:")
print(json.dumps({"type":"rollout","model":"both","episodes":[{"id":i,"seed":17} for i in chosen]}))
```
