Layer 13 head 5 has higher image-attention entropy in both recorded loops and substantial pose coupling. I will inspect its source distribution and sharpen only that head’s queries by 1.5. The probe must confirm that this concentrates visual evidence rather than simply increasing attention to text or self tokens.
```python
for i in range(4):
    a=maps[13][i][5].mean(0)
    top=np.argsort(a[:512])[-5:][::-1]
    print(i,"B/W/lang/self",np.round([a[:256].sum(),a[256:512].sum(),a[768:].sum(),1-a.sum()],3).tolist(),
          "top image patches",[(int(p//256),int((p%256)//16),int(p%16),round(float(a[p]),3)) for p in top])
n13=f"{expert}layers.13.self_attn.q_proj.weight"
qq=load_original(n13); qq[1280:1536]*=1.5; save_edit(n13,qq)
(W/"current_candidate.json").write_text(json.dumps({"name":"E13H5_Q_1.5","layer":13,"head":5,"query_gain":1.5}))
with (W/"notes.md").open("a") as f:
    f.write("\n## Spatial-concentration hypothesis\n"
            "- E13H5 conditional image-attention entropy is higher in both recorded loops (paired differences +0.953 and +0.243 nats).\n"
            "- Its gated, final-normalization-adjusted pose coupling is substantial (0.0383 energy).\n"
            "- Candidate query gain 1.5 for only E13H5, to concentrate current visual localization rather than rebalance cameras or weaken grip commands.\n")
```
