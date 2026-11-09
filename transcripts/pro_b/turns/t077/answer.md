The half-dose DC edit reduced loop-failure counts but did not improve environment success. I will test the full dose of the same rank-one edit, without changing its direction or calibration. This checks whether partial attenuation leaves too much brightness sensitivity; I will reject it if stronger invariance damages the controls or still fails to improve success.
```python
alpha=1.0
newW=(Pflat-alpha*dc_out[:,None]*color_v[None,:]).reshape_as(PW)
newB=load_original(patchb)+alpha*dc_out*mu_v
save_edit(patchw,newW); save_edit(patchb,newB)
dst=W/"candidate_patch_dc_full"
dst.mkdir(exist_ok=True)
for p in (W/"edited").iterdir():
    if p.is_file(): shutil.copy2(p,dst/p.name)
print("Full-dose relative patch-weight change",round(float((newW-PW).norm()/PW.norm()),4))
print("Remaining uniform-direction response / original",
      round(float((newW.reshape(1152,-1)@color_v).norm()/dc_out.norm()),7))
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 24: DC half-dose, inconclusive\n"
            "84/128->86/128 (+1.6 +/-2.2pp); environment44/80 unchanged, "
            "object20->21/24, semantic20->21/24. Loop failures31->24. "
            "Testing full removal of identical DC direction (no re-estimation), "
            "to establish a dose-response and test exact uniform-color invariance.\n")
```
