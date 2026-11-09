Channel 738 is strongly tied to BOS: its language average is approximately inversely proportional to prompt length, while its camera average is near zero. BOS also has an unusually large embedding component in channel 604. I will halve only these two columns of the VLM key projections in layers 3–12, preserving values and other channels, then check whether BOS attention actually falls.
```python
sink_layers=list(range(3,13))
sink_dims=[738,604]
for l in sink_layers:
    n=f"{lm}.layers.{l}.self_attn.k_proj.weight"
    a=load_original(n)
    a[:,sink_dims] *= 0.5
    save_edit(n,a)
dst=W/"candidate_bos_key_half"
dst.mkdir(exist_ok=True)
for p in (W/"edited").iterdir():
    if p.is_file(): shutil.copy2(p,dst/p.name)
# Quantify the BOS-like inverse-length signature without using success labels.
lens=[]
for x in pr[3][:30]:
    r=rec_for_path(x["input"])
    lens.append(1+len(sp.encode(r["instruction"]+"\n")))
lens=np.array(lens)
for d in sink_dims:
    vals=np.array([a["mean::vlm.5.resid::lang"][d] for a in act3[:30]])
    img=np.array([a["mean::vlm.5.resid::base"][d] for a in act3[:30]])
    print("channel",d, "corr with inverse length",round(float(np.corrcoef(vals,1/lens)[0,1]),3),
          "n*lang mean/std",round(float((vals*lens).mean()),1),round(float((vals*lens).std()),1),
          "image mean abs",round(float(np.abs(img).mean()),2))
inputs_sink=[x["input"] for x in pr[2]]
print("Probe inputs:",inputs_sink)
with (W/"notes.md").open("a") as f:
    f.write("\n## Candidate BOS-key attenuation\n"
            "Deep pooled language residual738 is approximately inverse prompt length "
            "(n*mean about-1800), camera residual738 near zero. BOS embedding738=-11.62 "
            "and604=-8.88 versus content-token magnitudes about0.3-0.6. "
            "Halve VLM k_proj columns738,604 in layers3-12 only, preserving values. "
            "Prediction: reduced BOS sink attention, more use of actual content/observations.\n")
(W/"current_candidate.json").write_text(json.dumps({"name":"bos_key_half","layers":sink_layers,"dims":sink_dims,"gain":0.5}))
```
