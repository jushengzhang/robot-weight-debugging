The later maps identify a more specific pattern: layer 20 head 15 puts roughly half its video attention on a single token and is enriched about 1.9-fold on texture-changed regions. Similar, weaker behavior appears in three layer-10 heads and one layer-14 head. I will soften only these five heads' query normalization by 35%, preserving their value and output projections, then check whether their attention becomes less texture-sensitive.
```python
texture_heads={10:[5,13,16],14:[18],20:[15]}
def apply_query_heads(dose=0.35, heads=None):
    if heads is None:
        heads=texture_heads
    clear_edits()
    report=[]
    for L,hs in heads.items():
        name=f"mot.mixtures.action.blocks.{L}.self_attn.norm_q.weight"
        original=load_original(name)
        x=original.clone()
        for h in hs:
            x[h*128:(h+1)*128] *= (1-dose)
        save_edit(name,x)
        report.append({"layer":L,"heads":hs,"relative_change":float((x-original).norm()/original.norm())})
    spec={"family":"texture_reader_query_temperature","heads":heads,"dose":dose,"tensors":report}
    (W/"current_candidate.json").write_text(json.dumps(spec,indent=2))
    return spec
qcandidate=apply_query_heads(.35)
print("Applied:",qcandidate)
# Preselect control heads from the same layers, using low background enrichment.
control_heads={}
for L,hs in texture_heads.items():
    H=late_map_stats[L].mean(0)
    qw=load_original(f"mot.mixtures.action.blocks.{L}.self_attn.norm_q.weight").reshape(24,128)
    qnorm=qw.norm(dim=1).numpy()
    onorm=load_original(f"mot.mixtures.action.blocks.{L}.self_attn.o.weight").reshape(1024,24,128).square().sum((0,2)).sqrt().numpy()
    available=[h for h in range(24) if h not in hs and H[h,1]<1.0 and H[h,0]>.65]
    chosen=[]
    for target in hs:
        options=[h for h in available if h not in chosen]
        def cost(h):
            return (math.log(qnorm[h]/qnorm[target])**2
                    +math.log(onorm[h]/onorm[target])**2
                    +(H[h,3]-H[target,3])**2)
        chosen.append(min(options,key=cost))
    control_heads[L]=chosen
    print("L",L,"target/control",hs,chosen,
          "control stats",[(h,np.round(H[h,:4],3).tolist()) for h in chosen])
(W/"query_temperature_targets.json").write_text(json.dumps({
    "targets":texture_heads,"controls":control_heads},indent=2))
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass17: late attention-concentration hypothesis\n"
            "- Replay audit: identical images/proprio but original actions differ by ~.001 RMS between runs. Small early probe effects were near numerical variability floor, not confirmed benefits.\n"
            "- Later maps: L20 H15 video mass .991, texture-region enrichment1.934, TV.435, peak .536. Also implicated L10 H5/H13/H16 and L14 H18 (consistent enrichment across two pairs).\n"
            "- New candidate: attenuate only these five heads' action norm_q slices by35%, softening texture-dependent attention concentration. All other parameters original. Preselected matched low-background control heads.\n"
            "- Passes used17/40; still no accepted edit.\n")
```