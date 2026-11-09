The unmasked tag contrast is strongly concentrated: one feature direction explains 73% of its energy. I will test a rank-one projection at the video text MLP's output, removing that direction while protecting the measured clean/paraphrase mean span and the linear proprioception span. This differs from the earlier input-space edits: it acts after the text MLP's nonlinearity, where the contrast is much more concentrated.
```python
# Static oblique projector: annihilate a contrast subspace while fixing protected directions.
_,su,vhu=torch.linalg.svd(diffU,full_matrices=False)
_,sm,vhm=torch.linalg.svd(diffM,full_matrices=False)
def make_oblique(B,C):
    E=B-C@(C.T@B)
    G=B.T@E
    D=E@torch.linalg.inv(G)
    assert float((D.T@B-torch.eye(B.shape[1],dtype=B.dtype)).abs().max())<1e-7
    return B,D
def apply_hidden_projector(rank=1,dose=1.0,mode="uniform"):
    clear_edits()
    vh={"uniform":vhu,"masked":vhm}[mode]
    B,D=make_oblique(vh[:rank].T,C)
    n=vd+"text_embedding.2.weight"
    orig=load_original(n).double()
    out=orig-dose*(orig@B)@D.T
    save_edit(n,out.float())
    print("VIDEO HIDDEN PROJECTOR",mode,"rank",rank,"dose",dose,
          "relative change",float((out-orig).norm()/orig.norm()),
          "clean mean error",float((cleanF@(out-orig).T).norm()),
          "basis conditioning",float(torch.linalg.cond(B.T@(B-C@(C.T@B)))))
    torch.save({"B":B,"D":D,"C":C,"rank":rank,"dose":dose,"mode":mode},W/"current_projector.pt")
apply_hidden_projector(1,1.0,"uniform")
with (W/"notes.md").open("a") as f:
    f.write("\nCandidate video_hidden_P1: one tensor video.text_embedding.2.weight right-composed with rank1 oblique projector. "
            "Uniform post-GELU contrast top mode73% energy; masked top mode47%. Protect79D measured clean/paraphrase means "
            "plus linear input-proprio/bias span; static SVD, no fitting. Must evaluate clean token effects empirically.\n")
```
