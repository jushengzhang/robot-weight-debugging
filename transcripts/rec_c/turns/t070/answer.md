VLM layer 0 attention is the main initial writer: its mean output in channel 674 is about -371, versus a total output norm of about 407. I will halve only that row of its output projection. The test is whether reducing this common offset makes visual differences more influential without broadly weakening attention.
```python
n=f"{vlmprefix}.layers.0.self_attn.o_proj.weight"
a=load_original(n);b=a.clone()
b[674]*=0.5
save_edit(n,b)
print("relative matrix change",round(float((b-a).norm()/a.norm()),5),
      "changed entries",int((a!=b).sum()),"/",a.numel())
print("backup",backup_edits("candidate_vlm0_sink674_half"))
with (W/"notes.md").open("a") as f:
    f.write("\n## VLM residual sink diagnosis and next causal test\n"
            "On32 initial observations, VLM residual mean-token energy in channel674 is93.2% atL2,89.1% atL8. "
            "LN gamma suppresses it but it still contributes~30-35% of normalized mean energy atL2/L8. "
            "Main initial writer is VLM L0 attention: mean channel674=-371 vs output norm407; "
            "its o_proj row674 norm17.45 vs row median1.24. "
            "Candidate halves only row674 of VLM L0 attention o_proj. "
            "Hypothesis: common-mode RMS bottleneck suppresses small visual differences; not established "
            "until fixed-input probe and paired recovery/ordinary-proxy tests.\n")
del a,b
```
