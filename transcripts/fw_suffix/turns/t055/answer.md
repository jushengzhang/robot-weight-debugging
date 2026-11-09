The context acquisition is complete. I will now remove every instrumentation edit and use a static SVD to identify tag-contrast directions orthogonal to the measured clean/paraphrase context means and the exact proprioception-encoder span.
```python
print(collect_context_slice(20,3584))
# Remove every temporary diagnostic intervention.
for n in list(edited()):
    drop_edit(n)
assert not edited()

X = torch.from_numpy(np.concatenate(
    [np.load(W/f"context_slice_{o:04d}.npy") for o in range(0,4096,512)], axis=1)).double()
np.save(W/"context_means_80x4096.npy", X.float().numpy())
Rctx, Cctx, Lctx = X[:48:2], X[1:48:2], X[48:]
P = torch.cat([load_original("proprio_encoder.weight").double(),
               load_original("proprio_encoder.bias").double()[:,None]], dim=1)
Cprotect = torch.cat([Cctx.T, Lctx.T, P], dim=1)
Qc, sc, _ = torch.linalg.svd(Cprotect, full_matrices=False)
D = (Rctx-Cctx).T
Dp = D-Qc@(Qc.T@D)
Utag, st, Vtag = torch.linalg.svd(Dp, full_matrices=False)
sraw = torch.linalg.svdvals(D)
print("Protected dimensions:", Qc.shape[1])
print("Protected-span singular values, smallest 8:", sc[-8:].tolist())
print("Tag contrast singular values:", np.round(sraw.numpy(),5).tolist())
print("Protected tag-contrast singular values:", np.round(st.numpy(),5).tolist())
print("Tag energy outside protected span:", float(Dp.square().sum()/D.square().sum()))
print("Cumulative residual contrast energy:",
      {k:round(float(st[:k].square().sum()/st.square().sum()),4) for k in [1,2,4,6,8,12,16,24]})
print("Protected overlaps:", float((Qc.T@Utag).abs().max()))
print("Context norms raw/clean/language:", *[round(float(a.norm(dim=1).mean()),4) for a in [Rctx,Cctx,Lctx]])
torch.save({"U":Utag.float(), "Qclean":Qc.float(), "singular_values":st.float(),
            "D":D.float(), "D_perp":Dp.float()}, W/"text_tag_basis.pt")
print("Original probe references:")
for p in [4,9,11]:
    rr=json.loads((W/f"probe_{p}"/"results.json").read_text())
    print(p, "n", len(rr), "first", {k:v for k,v in rr[0].items() if k!="activations"})
print("Current edited tensors:", edited())
with (W/"notes.md").open("a") as f:
    f.write("\n### Pass 20: full text-context acquisition complete\n"
            "Saved context_means_80x4096.npy and text_tag_basis.pt. ALL temporary instrumentation removed. "
            "Tag basis is SVD of paired context-mean differences after orthogonal removal of 24 clean means, "
            "32 paraphrase means, and the 9D proprio encoder weight+bias span. No optimization/training.\n")
```
