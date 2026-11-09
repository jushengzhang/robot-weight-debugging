I will quantify the text-neuron intervention, including first-ten-step arm actions and layerwise contrast changes. If it shows a favorable direction, I will test a stronger dose; otherwise I will discard the hypothesis rather than infer importance from weight norms alone.
```python
summary11=summarize_probe(11)
A11=probe_arrays(11)
print("Text-outlier half dose: arm RMS",round(rms(A11[0::2,:,:6]-base_clean[:,:,:6]),5),
      "clean arm drift",round(rms(A11[1::2,:,:6]-base_clean[:,:,:6]),5),
      "first10 arm RMS",round(rms(A11[0::2,:10,:6]-base_clean[:,:10,:6]),5))
p11=json.loads((work/"probe_11/results.json").read_text())
p11acts=[dict(np.load(r["activations"])) for r in p11]
print("Layer paired-contrast norm ratio versus original:")
for l in [1,7,12,23,29]:
    k=f"mean::action.{l}.attn_out::action"
    orig=np.stack([probe4_acts[2*i][k]-probe4_acts[2*i+1][k] for i in range(24)])
    new=np.stack([p11acts[2*i][k]-p11acts[2*i+1][k] for i in range(24)])
    print(l,round(float(np.linalg.norm(new)/np.linalg.norm(orig)),4))
with (work/"notes.md").open("a") as f:
    f.write("\n### Pass 11: text_outliers_2_half controlled probe\n"+json.dumps(summary11)+"\n")
```
