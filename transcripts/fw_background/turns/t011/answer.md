The FFN scan also shows no single pathological channel that warrants an edit by itself. I will prepare reversible edit and audit utilities now. The intended interventions are head-local or low-rank changes, with exact backups, rather than scaling an entire tensor.

```python
def note(text):
    with (W/"notes.md").open("a") as f:
        f.write("\n"+text.strip()+"\n")

def reset_edits():
    for n in list(edited()):
        drop_edit(n)

def snapshot_edits(label, meta=None):
    folder = W/"snapshots"/label
    folder.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i,n in enumerate(sorted(edited())):
        filename = f"tensor_{i:03d}.pt"
        torch.save(load(n).clone(), folder/filename)
        manifest.append({"name":n,"file":filename})
    (folder/"manifest.json").write_text(json.dumps(
        {"tensors":manifest,"metadata":meta or {}},indent=2))
    return str(folder)

def restore_snapshot(label):
    folder = W/"snapshots"/label
    manifest = json.loads((folder/"manifest.json").read_text())
    reset_edits()
    for rec in manifest["tensors"]:
        save_edit(rec["name"], torch.load(folder/rec["file"],map_location="cpu",weights_only=True))
    return manifest["metadata"]

def edit_action_output_heads(layer_heads, dose):
    """Attenuate only the listed 128-column head slices, starting from original."""
    reset_edits()
    for l,heads in layer_heads.items():
        n = pname("action",int(l),"self_attn.o.weight")
        a = load_original(n).clone()
        for h in heads:
            a[:,128*h:128*(h+1)] *= 1-dose
        save_edit(n,a)

def edit_action_output_subspaces(specs, dose):
    """Remove a recorded nuisance output direction only from specified head slices."""
    reset_edits()
    for l,head_to_basis in specs.items():
        n = pname("action",int(l),"self_attn.o.weight")
        a = load_original(n).clone()
        for h,basis in head_to_basis.items():
            u = torch.as_tensor(basis,dtype=torch.float32)
            if u.ndim==1:
                u = u[:,None]
            u = torch.linalg.qr(u,mode="reduced").Q
            sl = slice(128*h,128*(h+1))
            a[:,sl] -= dose*u@(u.T@a[:,sl])
        save_edit(n,a)

def audit_edits():
    report=[]
    for n in sorted(edited()):
        a,o=load(n),load_original(n)
        d=a-o
        report.append({"name":n,"shape":list(a.shape),"finite":bool(torch.isfinite(a).all()),
                       "changed_fraction":float((d!=0).float().mean()),
                       "relative_frobenius_change":float(d.norm()/o.norm()),
                       "delta_norm":float(d.norm())})
    return report

note("## Intervention plan (not yet applied)\n"
     "- Inspect fixed-input matched-task residual and attention outputs, then attention maps.\n"
     "- Candidate families: selective action self-attention output head attenuation; low-rank removal within implicated heads.\n"
     "- Use original-referenced edits, causal dose response, and matched control-head interventions.\n"
     "- Snapshot/audit utilities prepared. Still no edits and no charged forward passes.")
print("Utilities ready; edits:", audit_edits())
print("Practice tasks by suite:",
      {s:len(set(v["_task"] for v in bycat["Background Textures"] if v["suite"]==s))
       for s in sorted(set(v["suite"] for v in variants))})
```
