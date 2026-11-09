Most measured clean-context energy is already protected; the paraphrase drift is concentrated in one diagnostic example. I will test rank 256 to see whether removing more of the tagged-context contrast gives a stronger correction, while continuing to track that safety issue rather than assuming the larger edit is better.
```python
import shutil
def snapshot_candidate(label):
    dest=W/"candidates"/label
    dest.mkdir(parents=True,exist_ok=True)
    for n in edited():
        np.save(dest/(n+".npy"),load(n).numpy())
    shutil.copyfile(W/"current_candidate.json",dest/"candidate.json")
    return str(dest)
snapshot_candidate("oblique_r64_d1_both")
def apply_context_candidate(rank=64,dose=1.,kind="oblique",experts=("action","video"),basis=None):
    basis = proj if basis is None else basis
    for n in list(edited()):
        drop_edit(n)
    uu=basis["U"][:,:rank].double()
    bb=(basis["B_nearest"] if kind=="oblique" else basis["U"])[:,:rank].double()
    for ex in experts:
        n=f"mot.mixtures.{ex}.text_embedding.0.weight"
        w0=load_original(n).double()
        wn=(w0-dose*(w0@bb)@uu.T).float()
        save_edit(n,wn)
        print(ex,"relative_delta",round(float(torch.linalg.norm(wn.double()-w0)/torch.linalg.norm(w0)),5))
    cfg={"rank":rank,"dose":dose,"kind":kind,"experts":list(experts)}
    (W/"current_candidate.json").write_text(json.dumps(cfg,indent=2))
    return cfg
apply_context_candidate(256,1.0)
with (W/"notes.md").open("a") as f:
    f.write("\nPass19 planned: same oblique construction, rank256 dose1 both. Clean sample SVD: 99.99% energy in446 dimensions, so512-PC protection covers almost all acquired clean-context energy. Pass18 language drift concentrated in l11 (normalized RMS.135), evidence of missing token directions/nonlinear sensitivity rather than broad measured-context leakage. All edits always rebuilt from originals.\n")
```
