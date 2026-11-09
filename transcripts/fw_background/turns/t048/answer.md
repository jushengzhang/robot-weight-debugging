I will check whether head 10's texture sensitivity repeats, then attenuate only that head's outgoing action-attention columns by 50%. This leaves the video encoder, all other heads, and the action scale unchanged. The causal prediction is a smaller texture-induced action difference, followed by better paired rollout outcomes.
```python
p5 = json.loads((W/"probe_5/results.json").read_text())
map_records = p4+p5
all_head_stats = {}
for L in [0,1,2,5]:
    vals=[]
    for j in range(0,8,2):
        a,b=map_records[j:j+2]
        M=np.load(a["attention"][str(L)]).mean(1)
        N=np.load(b["attention"][str(L)]).mean(1)
        mass=(M.sum(-1)+N.sum(-1))/2
        P=M/M.sum(-1,keepdims=True).clip(1e-12)
        Q=N/N.sum(-1,keepdims=True).clip(1e-12)
        d=token_difference(a["input"],b["input"])
        roi=d>np.quantile(d,0.6)
        enrich=((P+Q)/2)[:,roi].sum(-1)/roi.mean()
        tv=np.abs(P-Q).sum(-1)/2
        vals.append(np.stack([mass,enrich,tv],1))
    all_head_stats[L]=np.stack(vals)
    H=np.mean(vals,axis=0)
    print("L",L,"top enrichment*shift heads:",
          [(int(h),np.round(H[h],3).tolist()) for h in np.argsort(-(H[:,1]*H[:,2]))[:5]])
print("L2 H10 by pair (video mass,enrichment,TV):", all_head_stats[2][:,10,:].round(3).tolist())
wname="mot.mixtures.action.blocks.2.self_attn.o.weight"
w0=load_original(wname)
norms=w0.reshape(1024,24,128).square().sum((0,2)).sqrt().numpy()
print("L2 head output norms:", np.round(norms,2).tolist())
print("L2 low-sensitivity candidates:",
      [(int(h),round(float(norms[h]/norms[10]),3),all_head_stats[2].mean(0)[h].round(3).tolist())
       for h in np.argsort(all_head_stats[2].mean(0)[:,1]*all_head_stats[2].mean(0)[:,2])[:8]])
def clear_edits():
    for n in list(edited()):
        drop_edit(n)
def apply_head_dose(layer, head, dose):
    clear_edits()
    name=f"mot.mixtures.action.blocks.{layer}.self_attn.o.weight"
    orig=load_original(name)
    x=orig.clone()
    x[:,head*128:(head+1)*128] *= (1.0-dose)
    save_edit(name,x)
    return {"family":"head_output_attenuation","layer":layer,"head":head,
            "dose":dose,"relative_frobenius":float((x-orig).norm()/orig.norm())}
candidate=apply_head_dose(2,10,0.50)
(W/"current_candidate.json").write_text(json.dumps(candidate,indent=2))
print("Applied:",candidate)
# Save compact input metadata for the ensuing common-noise comparisons.
fixed8=[r["input"] for r in map_records]
(W/"causal_probe_paths.json").write_text(json.dumps(fixed8))
for j in [0,2,4,6]:
    a=np.load(map_records[j]["action"]); b=np.load(map_records[j+1]["action"])
    print("Original pair RMS:",j//2,round(float(np.sqrt(np.mean((a-b)**2))),5))
with (W/"notes.md").open("a") as f:
    f.write("\n## Passes 4-5: early-head attention localization\n"
            "- Four matched texture pairs, common noise 17, layers 0/1/2/5.\n"
            "- Testing action layer 2 head 10: high video reliance, texture-region enrichment and spatial attention shift. Correlation is a candidate screen, not causal evidence.\n"
            "- First causal edit: only L2 H10 output-projection columns attenuated by 50%; all other tensors unchanged.\n"
            "- Passes used 5/40. This edit is experimental, not accepted.\n")
```