Several mid-layer heads are also strongly low-rank, so I will not limit the diagnosis to the final layer. I will compare the dominant query/key directions with their biases and record candidate directions for later probes. This can help distinguish fixed attention preferences from content-dependent sensitivity once observations are available.```python
ranked = np.argsort(spectral_stats["action_q_top4"].ravel())[-20:][::-1]
weight_candidates=[tuple(map(int,np.unravel_index(i,(30,24)))) for i in ranked]
bias_alignment=[]
for l,h in weight_candidates:
    d={"layer":l,"head":h}
    for ex,p in (("action","q"),("video","k")):
        mat=load_original(nm(ex,l,f"self_attn.{p}.weight"))[h*128:(h+1)*128]
        bias=load_original(nm(ex,l,f"self_attn.{p}.bias"))[h*128:(h+1)*128]
        gain=load_original(nm(ex,l,f"self_attn.norm_{p}.weight"))[h*128:(h+1)*128]
        gram=(mat*gain[:,None])@(mat*gain[:,None]).T
        ev,U=torch.linalg.eigh(gram)
        u=U[:,-4:]
        gb=gain*bias
        d[ex+"_"+p+"_bias_top4"]=round(float((u.T@gb).square().sum()/gb.square().sum().clamp_min(1e-12)),4)
        d[ex+"_"+p+"_bias_rms"]=round(float(gb.square().mean().sqrt()),4)
        mass=u.square().sum(1)
        d[ex+"_"+p+"_coordinate_bands"]=np.round(
            [float(mass[:44].sum()/4),float(mass[44:86].sum()/4),float(mass[86:].sum()/4)],3).tolist()
        del mat,bias,gain,gram,ev,U,u,gb,mass
    bias_alignment.append(d)
print("Dominant-direction bias alignment (top candidate heads):")
for d in bias_alignment[:12]:
    print(d)
(W/"weight_bias_alignment.json").write_text(json.dumps(bias_alignment,indent=2))
print("Candidate layer/head list:",weight_candidates)
with open(W/"notes.md","a") as f:
    f.write("\n## Weight-only observations (not yet causal)\n"
            "- Action-q vs video-k per-head RMS normalization gains correlate 0.9982.\n"
            "- Layer 29 heads 4/7/8/17 have sparse normalization outliers up to 7.625.\n"
            "- Several mid-layer heads also have strongly concentrated query spectra, "
            "including L12H21, L16H13, L18H7, L23H3; avoid selecting from norms alone.\n"
            "- No edits applied; simulator still offline at the last request.\n")
```
