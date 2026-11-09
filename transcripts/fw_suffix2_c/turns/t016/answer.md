I will quantify the baseline action gap and its layer-by-layer growth, then make a temporary causal edit that removes text-specific input to the video expert while exactly preserving the proprio encoder's span. This is a localization test, not yet a repair: it asks how much corruption reaches actions through the observation encoder.
```python
p5 = json.loads((W/"probe_5"/"results.json").read_text())
print("RESULT FORMAT:",str(p5)[:1700])
print("ACT FILES EXAMPLE:")
first5 = p5[0] if isinstance(p5,list) else p5
print(type(p5).__name__)
print("FILES:",[p.name for p in (W/"probe_5").iterdir()][:12])
# Extract the previous detector without applying its gate.
gate_meta={}
for ex in ("action","video"):
    pref=f"mot.mixtures.{ex}."
    ew=torch.from_numpy(np.load(W/"prior_candidate"/(pref+"text_embedding.0.weight.npy")))
    eb=torch.from_numpy(np.load(W/"prior_candidate"/(pref+"text_embedding.0.bias.npy")))
    d=ew-load_original(pref+"text_embedding.0.weight")
    ix=int(d.norm(dim=1).argmax())
    gate_meta[ex]={"hidden_neuron":ix,"detector":ew[ix]/32.0,"threshold":float(-eb[ix]/32.0)}
print("PRIOR DETECTOR:",{k:{"neuron":v["hidden_neuron"],"norm":float(v["detector"].norm()),"threshold":v["threshold"]} for k,v in gate_meta.items()})
print("DETECTOR AGREEMENT:",float(torch.nn.functional.cosine_similarity(gate_meta["action"]["detector"],gate_meta["video"]["detector"],dim=0)))
Pspan=torch.cat([load_original("proprio_encoder.weight"),load_original("proprio_encoder.bias")[:,None]],dim=1)
Qp=torch.linalg.qr(Pspan,mode="reduced").Q
print("PROPRIO SPAN RANK:",Qp.shape[1],"detector leakage",float((Qp.T@gate_meta["action"]["detector"]).norm()))
def clear_edits():
    for n in list(edited()):
        drop_edit(n)
def text_dose(ex, dose):
    n=f"mot.mixtures.{ex}.text_embedding.0.weight"
    orig=load_original(n)
    proj=(orig@Qp)@Qp.T
    save_edit(n,orig-dose*(orig-proj))
clear_edits()
text_dose("video",1.0)
print("TEMPORARY EDIT:",edited())
with (W/"notes.md").open("a") as f:
    f.write("\nPass5: original common-noise reference on272 inputs saved. Next diagnostic: video text-entry projection "
            "onto exact9D proprio weight+bias span (one tensor); removes text-specific input. This is not a selected repair.\n")
```
