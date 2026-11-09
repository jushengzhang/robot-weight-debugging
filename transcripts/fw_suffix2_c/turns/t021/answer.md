I will measure the action-only ablation and restore the original weights. I will also compare the first-block residuals across the two interventions to distinguish direct injection from later propagation through shared attention.
```python
m7=screen_metrics(7)
clear_edits()
for num in (5,6,7):
    if num==5:
        rows=[orig_by_input[str(p.resolve())] for p in screen_sources]
    else:
        rows=json.loads((W/f"probe_{num}"/"results.json").read_text())
    print("PASS",num,"RESID TAG GAP (first denoising call / episode-mean):")
    for layer in (0,1,2,5,10,15,20,25,29):
        key=f"calls::action.{layer}.resid::action"
        x=np.stack([np.load(r["activations"])[key] for r in rows[:96]])
        print(layer,round(rms(x[:48,0]-x[48:96,0]),5),round(rms(x[:48].mean(1)-x[48:96].mean(1)),5),end="; ")
    print()
print("ALL EDITS CLEARED:",len(edited()))
```
