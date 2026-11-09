The combined gate reduces the action gap by only 3.2%, so direct suppression of those context responses explains only a small part of the problem. I will now test the oblique projection at both text-entry layers, removing all 24 measured contrast modes while preserving the same clean/paraphrase and proprioception span.
```python
def apply_oblique_text(rank=24,strength=1.,experts=("action","video")):
    for n in list(edited()):
        drop_edit(n)
    basis=torch.load(W/"text_oblique_basis.pt",map_location="cpu",weights_only=True)
    u=basis["U"][:,:rank]; b=basis["B"][:,:rank]
    changes={}
    for exp in experts:
        n=f"mot.mixtures.{exp}.text_embedding.0.weight"
        w=load_original(n)
        dw=strength*((w@b)@u.T)
        save_edit(n,w-dw)
        changes[exp]=float(dw.norm()/w.norm())
    state={"name":"oblique_text_contrast_removal","rank":rank,"strength":strength,
           "experts":list(experts),"relative_changes":changes,"edited_names":edited()}
    (W/"current_edit.json").write_text(json.dumps(state,indent=2))
    return state
print(json.dumps(apply_oblique_text(),indent=2))
with (W/"notes.md").open("a") as f:
    f.write("\n### Candidate: rank-24 oblique text contrast removal\n"
            "Only action/video text_embedding.0.weight changed. P=I-BU^T, U^T B=I. "
            "B contains full paired contrast modes, U their clean/proprio-orthogonal detectors. "
            "This is an explicit idempotent geometric projector, no loss fitting or optimization. "
            "Both attention-gate candidates retained as reproducible metadata, not combined with this test.\n")
```
