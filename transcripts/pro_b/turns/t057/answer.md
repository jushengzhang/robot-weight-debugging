The algebraic shifts can preserve expert keys and values exactly in float32 while adding one nominal logit to the selected head’s non-BOS language direction. I will apply that conservative dose through the adaptive normalization’s shift bias at layers 6 and 12. A probe will determine whether the pooled-key approximation transfers to actual token attention.
```python
for l,shift in qshift.items():
    n=f"{ex}.layers.{l}.input_layernorm.dense.bias"
    b=load_original(n)
    b[1024:2048] += shift
    save_edit(n,b)
dst=W/"candidate_query_content_d1"
dst.mkdir(exist_ok=True)
for p in (W/"edited").iterdir():
    if p.is_file(): shutil.copy2(p,dst/p.name)
with (W/"notes.md").open("a") as f:
    f.write("\n## Candidate content-query shift\n"
            "Two expert adaptive input-norm shift biases (L6,L12), "
            "computed by a direct nullspace projection. Preserve expert K/V, "
            "pooled camera means/PCs and BOS key response; target H2 non-BOS "
            "language +1 nominal logit using slow RoPE dimensions. "
            "Shift RMS .121/.161. No success labels, gradients or action targets used. "
            "Other-head pooled language response constrained to zero.\n")
print("Active edits:",edited())
print("Probe inputs:",inputs_sink)
```
