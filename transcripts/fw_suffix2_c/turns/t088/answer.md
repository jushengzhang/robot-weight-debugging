The new clean-span estimate was smaller than the one used in the earlier low-drift test. I’ll retain that larger protected span so this comparison changes the contamination directions, not the amount of clean-language protection.
```python
oldp=torch.load(W/"weighted_video_projector.pt",weights_only=False)
C=orth_columns(torch.cat([oldp["C"].double(),excl],dim=1),rtol=2e-6)
ub,sb,_=torch.linalg.svd(dm,full_matrices=False)
B0=ub[:,:128]
ur,sr,vhr=torch.linalg.svd(B0-C@(C.T@B0),full_matrices=False)
keep=sr>=0.15
B=(B0@vhr.T[:,keep]).float()
D=(ur[:,keep]/sr[keep]).float()
proj={"B":B,"D":D,"C":C.float(),"angles":sr.float(),
      "contrast_rank":128,"clean_energy":"previous protected span","min_angle":0.15}
torch.save(proj,W/"masked_video_projector128.pt")
changed=orig-(orig@B)@D.T
save_edit(name,changed)
np.save(dest/(name+".npy"),changed.numpy())
print("retained",B.shape[1],"protected",C.shape[1],
      "relative_change",float((changed-orig).norm()/orig.norm()))
print("screen request inputs:",json.dumps(screen_paths))
with (W/"notes.md").open("a") as f:
    f.write(f"\nBefore probe29, strengthened protection to the previous weighted-video protected span plus unread channels: {C.shape[1]}D, {B.shape[1]} retained masked contrast directions. No results used for this adjustment.\n")
```
