The layer 10 intervention changed attention as intended but reduced screening success by 4.7 points, so I will not retain it. I will make one more localized routing test at layer 12's object-focused spatial head, whose top base-image patches differ from the earlier head's. This tests whether the useful observation signal is being lost at a different point, rather than applying a broad camera rebalance.
```python
print("Pass 9:",paired_summary(9))
for ii in [0,1,2,3]:
    print("Input",ii,"top image patches")
    for l,h,aa in [(8,7,att2),(10,1,att3),(12,3,att2),(13,3,att5)]:
        A=aa[ii][l][h]
        ix=np.argsort(A[:512])[::-1][:4]
        print(f" L{l}H{h}",[("B" if j<256 else "W",int(j%256//16),int(j%16),round(float(A[j]),3)) for j in ix])
clear_edits()
cn=camera_query_edit(12,3,1.0)
print("Current edit:",cn,"relative change",round(float((load(cn)-load_original(cn)).norm()/load_original(cn).norm()),5))
with (W/"notes.md").open("a") as f:
    f.write("\n## Passes 8-9: L10 H1 camera-direction removal rejected\n"
            "- Probe: base mass in two loops rose 0.041->0.275 and 0.034->0.180; "
            "base-focused success 0.701->0.691. Action changes tiny (XYZ RMS ~0.001).\n"
            "- Paired screen 31/64 -> 28/64, -4.7 pp (SE 3.5), 1 rescue / 4 losses; loops 21->22.\n"
            "- Reverted; next isolated test is the same directional edit in spatial L12 H3.\n")
```
