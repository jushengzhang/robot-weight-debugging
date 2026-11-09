I’ll assemble the positive-minus-negative summaries, verify their pairing, and remove the instrumentation. Then I’ll use the known text-entry matrix to recover the observable part of the frozen text context, avoiding a comparison that is distorted by unequal channel scales.
```python
neg2=collect_negative_readout(33)
clear_edits()
neg=torch.zeros_like(M)
for j in range(3):
    ids=spec["feature_ids"][j*spec["m"]:(j+1)*spec["m"]]
    neg[:,:,ids]=torch.from_numpy(np.load(W/f"negative_video_{j}.npy"))
Z=M-neg
torch.save(Z.float(),W/"linear_video_features.pt")
for k in [31,32,33]:
    rr=json.loads((W/f"probe_{k}/results.json").read_text())
    assert [a["input"] for a in rr]==expected
print("active edits",len(edited()),"linear summaries",tuple(Z.shape))
print("clean uniform masked/unmasked RMS",
      float((Z[48:,0]-Z[48:,13]).square().mean().sqrt()))
ids=spec["feature_ids"]
A=load_original("mot.mixtures.video.text_embedding.0.weight")[ids].double()
bb=load_original("mot.mixtures.video.text_embedding.0.bias")[ids].double()
ua,sa,vha=torch.linalg.svd(A,full_matrices=False)
print("entry singular values", [round(float(sa[i]),6) for i in [0,1,8,128,1024,2048,-1]],
      "condition",float(sa[0]/sa[-1]))
# Coordinates of each observed context mean in the orthonormal row space of A.
coords=(Z[:,:,ids]-bb)@ua/sa
torch.save({"coords":coords.float(),"basis":vha.T.float(),"singular_values":sa.float(),
            "feature_ids":ids},W/"linear_context_coordinates.pt")
dd=(coords[:48]-coords[48:96])
ss=torch.linalg.svdvals(dd.reshape(-1,coords.shape[-1]).T)
cc=torch.linalg.svdvals(coords[48:].reshape(-1,coords.shape[-1]).T)
print("contrast energy",{r:round(float(ss[:r].square().sum()/ss.square().sum()),6)
                         for r in [1,4,16,32,64,128,256]})
print("clean uncentered ranks",
      {en:int(torch.searchsorted(cc.square().cumsum(0)/cc.square().sum(),en))+1
       for en in [.999,.9999,.99999,.999999]})
with (W/"notes.md").open("a") as f:
    f.write("\nPass33 negative readouts complete; all188 diagnostic tensors cleared. linear_video_features.pt and linear_context_coordinates.pt saved. Observable UMT5 row-space coordinates reconstructed algebraically with SVD of original video text-entry matrix; no local network evaluation or fitting.\n")
```
