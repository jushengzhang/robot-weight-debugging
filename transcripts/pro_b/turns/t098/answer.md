The restoration issue is confined to the analysis copies of two biases: all 102 repeated original-probe comparisons are bit-identical, and no other saved candidate matches its original tensor. The forward baseline is therefore intact. I will next test routing rather than value strength at layer 14, amplifying the controlled visual-language subspace in keys while explicitly protecting the two BOS-dominated coordinates.
```python
def original_clone(n):
    if n in clean_shift_bias:
        return clean_shift_bias[n].clone()
    return load_original(n).clone()
for n in list(edited()):drop_edit(n)
# Remove the BOS-coordinate overlap, then re-orthogonalize the same measured subspace.
U0=bases[14].clone()
U0[[604,738],:]=0
Uk=torch.linalg.qr(U0,mode="reduced").Q
kn=LP+"layers.14.self_attn.k_proj.weight"
K=original_clone(kn)
Ke=K+(K@Uk)@Uk.T
save_edit(kn,Ke)
cand=W/"candidate_controlled_visualK_g2"
cand.mkdir(exist_ok=True)
torch.save(Ke,cand/(kn+".pt"))
torch.save(Uk,W/"subspaces"/"controlled_grounded_key_basis.pt")
print("Key relative change",round(float((Ke-K).norm()/K.norm()),4),
      "protected column max diff",float((Ke-K)[:,[604,738]].abs().max()))
# Select interpretable already-studied task and semantic observations plus a position failure.
r1=[json.loads(x) for x in (W/"roll_1.jsonl").read_text().splitlines()]
posrec=next(r for r in r1 if r["ptype"]=="position" and not r["success"] and r["suite"]=="libero_object")
ppos=sorted(local_path(posrec["frames_dir"]).glob("t0000_inputs.pt"))[0]
probe29inputs=[inp_list[0],inp_list[1],inp_list[5],str(ppos)]
(W/"grounded_key_probe_inputs.json").write_text(json.dumps(probe29inputs))
print("Probe inputs",json.dumps(probe29inputs))
print("Position episode",posrec["id"],posrec["instruction"],"moved",posrec["obj_moved_m"])
with (W/"notes.md").open("a") as f:
    f.write("\n## Weight-copy audit and controlled-key candidate\n"
            "load_original analysis copies of expert adaptive-norm biases6/12 equal the old diagnostic query-shift edits. "
            "Recovered clean copies by subtracting saved shifts; do not mutate load_original returns. "
            "102 repeated original-action probe comparisons are bit-identical; forward baseline unaffected. No other candidate exactly matches originals. "
            "Next candidate: VLM14 k_proj rank8 visual-language gain2, using pure controlled-swap basis with BOS coordinates604/738 removed. "
            "Tests routing rather than retrieved value amplitude.\n")
```
