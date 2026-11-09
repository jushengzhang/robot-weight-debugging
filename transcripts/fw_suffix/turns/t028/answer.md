I will evaluate the readout projection and compare it with all earlier interventions using the executed first ten actions as well as the full chunk. This will determine whether it warrants a rollout test or should be discarded like the broader head ablations.
```python
summary10=summarize_probe(10)
print("Candidate comparison (RMS; normalized model actions):")
print("pass  raw6   clean6  raw10x6  clean10x6  raw10x7")
for pn in [4,5,6,7,8,10]:
    A=probe_arrays(pn); R,C=A[0::2],A[1::2]
    print(pn, *(f"{x:.5f}" for x in [
        rms(R[:,:,:6]-base_clean[:,:,:6]),
        rms(C[:,:,:6]-base_clean[:,:,:6]),
        rms(R[:,:10,:6]-base_clean[:,:10,:6]),
        rms(C[:,:10,:6]-base_clean[:,:10,:6]),
        rms(R[:,:10]-base_clean[:,:10])]))
with (work/"notes.md").open("a") as f:
    f.write("\n### Pass 10: action_head_P8 controlled probe\n"+json.dumps(summary10)+"\n")
print("Current candidate:",(work/"current_edit.json").read_text())
```
