The peak inspection exposes a confound: layer 13 head 5 repeatedly selects the same border patch, so its entropy difference is not convincing evidence of grasp-location averaging. I will undo that untested edit and inspect the other spatial candidates for genuinely observation-dependent patch locations.
```python
for n in list(edited()):drop_edit(n)
for L,h in [(8,7),(9,1),(12,0),(11,0),(12,2),(10,7),(12,3)]:
    print("Head",L,h)
    for i in range(4):
        a=maps[L][i][h].mean(0)
        top=np.argsort(a[:512])[-3:][::-1]
        print(i,"B/W/lang/self",np.round([a[:256].sum(),a[256:512].sum(),a[768:].sum(),1-a.sum()],3).tolist(),
              "patches",[(int(p//256),int((p%256)//16),int(p%16),round(float(a[p]),3)) for p in top])
with (W/"notes.md").open("a") as f:
    f.write("- Before any forward test, rejected E13H5 spatial interpretation: top image patch was wrist row0/col1 in all four inputs, likely a global/sink feature. Untested edit removed.\n")
(W/"current_candidate.json").write_text(json.dumps({"name":"none"}))
```
