The action-only gate barely changes the final tag gap, despite preserving clean actions closely. I will check a different possibility: the harmless part of the tag contrast lies outside the clean span, while its harmful component overlaps legitimate language directions. An oblique projection can use the former as a detector to remove the full paired contrast while still preserving the measured clean span.
```python
# Static geometric construction, not optimization: P = I - B U^T, with U^T B = I.
# It removes the full contrast modes detected by U, including their clean-span components.
Btag = (D @ Vtag.T) / st[None,:]
print("Oblique projector idempotence check:",float((Utag.T@Btag-torch.eye(24,dtype=torch.double)).abs().max()))
print("Protected-span invariance check:",float((Utag.T@Qc).abs().max()))
for exp in ["action","video"]:
    w=load_original(f"mot.mixtures.{exp}.text_embedding.0.weight").double()
    for k in [1,4,8,24]:
        orth=(w@Utag[:,:k])@Utag[:,:k].T
        ob=(w@Btag[:,:k])@Utag[:,:k].T
        print(exp,k,"orth_rel",round(float(orth.norm()/w.norm()),5),
              "oblique_rel",round(float(ob.norm()/w.norm()),5),
              "oblique/orth",round(float(ob.norm()/orth.norm()),3))
torch.save({"U":Utag.float(),"B":Btag.float()},W/"text_oblique_basis.pt")
print("Detector threshold boundary:")
for t in [.125,.25,.375,.5]:
    ix=features.index((0,t,32.))
    print(t,"raw min/mean",float(G[raw_ix,ix].min()),float(G[raw_ix,ix].mean()),
          "clean max",float(G[clean_ix,ix].max()),"clean min",float(G[clean_ix,ix].min()))
# Retain a reproducible record of the action-only gate, without treating it as validated.
(W/"candidate_action_gate.json").write_text((W/"current_edit.json").read_text())
```
