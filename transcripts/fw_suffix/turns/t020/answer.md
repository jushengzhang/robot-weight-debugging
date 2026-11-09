I will check whether the action cross-attention projection helped, separating arm-control differences from the gripper channel. For the next pathway test, I will select a small subset of video cross-attention heads by how strongly their output weights can write the observed nuisance directions relative to protected clean directions.
```python
summary6=summarize_probe(6)
for pass_no in [4,5,6]:
    A=probe_arrays(pass_no)
    print("Pass",pass_no,
          "arm raw->orig-clean",round(rms(A[0::2,:,:6]-base_clean[:,:,:6]),5),
          "arm clean drift",round(rms(A[1::2,:,:6]-base_clean[:,:,:6]),5),
          "first10 raw->orig-clean",round(rms(A[0::2,:10]-base_clean[:,:10]),5))

# Rank head groups using output-space nuisance vs clean-direction selectivity.
head_rankings={}
for expert in ["video","action"]:
    for l in range(30):
        U,Q,D,C,st=basis_data[(expert,l)]
        W=load_original(f"mot.mixtures.{expert}.blocks.{l}.cross_attn.o.weight")
        tag=(U.T@W).reshape(U.shape[1],24,128).square().sum((0,2))
        clean=(Q.T@W).reshape(Q.shape[1],24,128).square().sum((0,2))
        score=tag/(clean+1e-12)
        head_rankings[(expert,l)]=score.argsort(descending=True).tolist()

def attenuate_cross_heads(expert,layers,nheads=6,scale=0.0):
    changed=total=0.
    for l in layers:
        name=f"mot.mixtures.{expert}.blocks.{l}.cross_attn.o.weight"
        W=load_original(name)
        Wnew=W.clone()
        heads=head_rankings[(expert,l)][:nheads]
        for h in heads:
            Wnew[:,128*h:128*(h+1)]*=scale
        save_edit(name,Wnew)
        changed+=float((Wnew-W).square().sum())
        total+=float(W.square().sum())
    return math.sqrt(changed/total)

clear_edits()
head_delta=attenuate_cross_heads("video", range(30), nheads=6, scale=0.)
config={"name":"video_cross_H6","module":"video.cross_attn.o.weight",
        "layers":list(range(30)),"nheads":6,"scale":0.,
        "selection":"tag/clean output-direction weight energy"}
(work/"current_edit.json").write_text(json.dumps(config,indent=2))
(work/"head_rankings.json").write_text(json.dumps(
    {f"{e}.{l}":hs for (e,l),hs in head_rankings.items()},indent=2))
print("Video selected heads (examples):",
      {l:head_rankings[("video",l)][:6] for l in [0,7,10,12,23,29]})
print("Edit relative norm:",round(head_delta,4),"tensors:",len(edited()))
with (work/"notes.md").open("a") as f:
    f.write("\n### Pass 6: action_cross_P4 controlled probe\n"+json.dumps(summary6)+"\n"
            "\n### Candidate video_cross_H6\n"
            "Zero 6/24 video cross-attention output-head column groups in each layer; "
            "ranked by weight energy writing clean-protected contrast directions divided by "
            "energy writing clean directions. Biases untouched. This is a targeted-head "
            "causal screen, not evidence that those heads directly attend to tags. "
            "No self-attention or action weights edited.\n")
```
