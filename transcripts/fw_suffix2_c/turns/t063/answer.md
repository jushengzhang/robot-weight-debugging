The video projection retains 118 contrast directions while protecting 769 measured directions. I’ll test it alone first; combining it with the action edit now would make it harder to identify which route helps or causes damage.
```python
def apply_projector(factors,dose=1.0):
    expert=factors["expert"]
    n=f"mot.mixtures.{expert}.text_embedding.2.weight"
    w=load_original(n).double()
    b,d=factors["B"],factors["D"]
    out=w-dose*(w@b)@d.T
    save_edit(n,out.float())
    print("APPLIED",expert,"rank",b.shape[1],"dose",dose,"relative",float((out-w).norm()/w.norm()))
clear_edits()
apply_projector(wv,1.0)
snapshot("candidate_video_weighted_full")
with (W/"notes.md").open("a") as f:
    f.write("\nCandidate weighted_video_full: one video.text_embedding.2.weight, contrast top128 modes ->118 principal-angle-retained directions, clean protection769D, min-angle.15, dose1. Relative weight change.5095. Probe22 pending.\n")
```
