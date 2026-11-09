I will test a rank-four projection at the text entry point of both experts: \(W' = W - WUU^\top\). This removes the selected tag-contrast directions before the text MLP, while leaving its biases, all transformer blocks, and the proprioception pathway unchanged.
```python
def apply_text_projection(rank=4, strength=1.0, experts=("action","video")):
    for n in list(edited()):
        drop_edit(n)
    U = torch.load(W/"text_tag_basis.pt", map_location="cpu", weights_only=True)["U"][:,:rank]
    changes = {}
    for expert in experts:
        n = f"mot.mixtures.{expert}.text_embedding.0.weight"
        w = load_original(n)
        delta = strength*((w@U)@U.T)
        save_edit(n, w-delta)
        changes[expert] = float(delta.norm()/w.norm())
    state={"name":"text_entry_projection", "rank":rank, "strength":strength,
           "experts":list(experts), "relative_changes":changes, "edited_names":edited()}
    (W/"current_edit.json").write_text(json.dumps(state, indent=2))
    return state

def probe_actions(pass_id, field="action_norm"):
    records = json.loads((W/f"probe_{pass_id}"/"results.json").read_text())
    return np.stack([np.load(r[field]) for r in records]), records

Aref, ref_records = probe_actions(21)
def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a,dtype=np.float64)**2)))

def probe_metrics(pass_id):
    a, rr = probe_actions(pass_id)
    assert [r["input"] for r in rr] == [r["input"] for r in ref_records]
    r,c,l = a[:48:2],a[1:48:2],a[48:]
    br,bc,bl = Aref[:48:2],Aref[1:48:2],Aref[48:]
    target = bc-br
    change = r-br
    out={"pass":pass_id, "baseline_tag_gap":rms(br-bc),
         "raw_to_original_clean":rms(r-bc),
         "gap_ratio":rms(r-bc)/rms(br-bc),
         "edited_raw_clean_gap":rms(r-c),
         "clean_drift":rms(c-bc), "language_drift":rms(l-bl),
         "raw_change":rms(change),
         "correction_alignment":float(np.sum(change*target)/
              (np.linalg.norm(change)*np.linalg.norm(target)+1e-12)),
         "fraction_corrected":float(np.sum(change*target)/(np.sum(target*target)+1e-12))}
    out["by_category"]={}
    for cat in sorted(set(x["category"] for x in dm)):
        ix=[i for i,x in enumerate(dm) if x["category"]==cat]
        out["by_category"][cat]={"gap_ratio":rms((r-bc)[ix])/rms((br-bc)[ix]),
                                "clean_drift":rms((c-bc)[ix])}
    (W/f"probe_{pass_id}"/"metrics.json").write_text(json.dumps(out,indent=2))
    return out

print("Reference tag gap:", rms(Aref[:48:2]-Aref[1:48:2]))
a4,_=probe_actions(4)
print("Reference vs earlier original probe:", rms(Aref[:48]-a4))
print(json.dumps(apply_text_projection(4,1.0), indent=2))
with (W/"notes.md").open("a") as f:
    f.write("\n### Pass 21: matched original reference\n"
            "80-input reference includes all raw/clean pairs and 32 paraphrases, noise_seed=0. "
            "Candidate text-entry rank-4 full projection applied to action/video text_embedding.0.weight only; "
            "biases and all other tensors original. Protects measured clean/paraphrase means and exact proprio span, "
            "not necessarily every clean token.\n")
```
