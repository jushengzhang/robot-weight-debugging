Channel 1938 accounts for a large fraction of the late video contrast, and the strongest identified readout is action head 17 at layer 24. I will test the largest single text-write source feeding that route: the leading mode of video layer 23 cross-attention head 3. Halving only this mode preserves the head’s remaining value-to-output spectrum and avoids deleting the channel throughout the network.
```python
templates=torch.load(W/"rank_one_intervention_templates.pt",map_location="cpu",weights_only=False)
def set_template_edit(key,alpha,mode=0,label=None):
    clear_edits()
    t=templates[key]
    n=t["name"];h=t["head"];f=t["factors"][mode]
    original=load_original(n)
    new=original.clone()
    delta=-alpha*torch.outer(f["u"],f["r"])
    new[:,h*128:(h+1)*128]+=delta
    save_edit(n,new)
    meta={"label":label or f"{key}_mode{mode}_alpha{alpha}",
          "key":key,"mode":mode,"alpha":alpha,"name":n,
          "relative_delta":float(delta.norm()/original.norm()),
          "singular_before":f["s"],"singular_retained":(1-alpha)*f["s"]}
    (W/"active_candidate.json").write_text(json.dumps(meta,indent=2))
    return meta
print(set_template_edit("video_L23_H3",0.5))
with (W/"notes.md").open("a") as f:
    f.write("\n## Video text-write / action-read candidate\n"
            "- Episode-averaged same-task video contrasts have channel-1938 energy fractions L13 .384, L20 .498, L22 .724, L23 .472. Correlation is trajectory-confounded. In 128 discovery episodes L23 agent-channel mean is 59.29 (success) vs 52.11 (failure); prompt-length correlation -.241.\n"
            "- Static route: video L23 cross-attn H3 OV leading singular value 81.91 (97.65% energy), left mode |u1938|=.9964. Next layer strongest value read is H17, 1.65x random-direction sensitivity.\n"
            "- Candidate is source-local rank-one half attenuation of video L23 cross_attn.o H3, NOT global channel ablation. Preserves other effective OV modes. Awaiting causal rollout.\n")
```
