The second independent batch is also negative, bringing the pooled result to eight fewer successes in 256 episodes. I’m rejecting this candidate for futility and revising the remaining-pass plan. I’ll test one simpler alternative: remove the full tag-prototype components at the video text entry, rather than only their orthogonal residuals. The two validation batches already seen will not be used to validate the replacement.
```python
s37=paired_rollout(37)
pool37=paired_rollout([36,37])
cats=sorted(set(m[1] for m in geo["novel_meta"]))
B=torch.stack([geo["novel"][[i for i,m in enumerate(geo["novel_meta"]) if m[1]==c]].mean(0)
               for c in cats],dim=1).double()
QP=geo["Qprotect"].double()
D=B-QP@(QP.T@B)
QD=torch.linalg.qr(D,mode="reduced").Q
R=QD.T@B
AA=B@torch.linalg.inv(R)
print("Rejected independent gate:",pool37)
print("Oblique projector condition:",float(torch.linalg.cond(R)))
print("Idempotence residual:",float((QD.T@AA-torch.eye(6,dtype=torch.float64)).norm()))
print("Clean protection leakage:",float((QD.T@QP).norm()))
for n in list(edited()):
    drop_edit(n)
n=V+"text_embedding.0.weight"
ow=load_original(n)
ew=ow.double()-(ow.double()@AA)@QD.T
save_edit(n,ew.float())
for label,arr in [("clean+language",X[24:].reshape(-1,4096)),
                  ("matched-prefix",geo["raw_prefix"]),("novel",geo["novel"])]:
    coeff=arr.double()@QD
    removed=coeff@AA.T
    print(label,"relative context change",round(float(removed.norm()/arr.double().norm()),5))
torch.save({"B":B.float(),"A":AA.float(),"detector_basis":QD.float(),
            "categories":cats,"rank":6,"expert":"video"},
           W/"full_tag_prototype_projector.pt")
np.save(W/"candidate_video_full_tag_rank6.npy",ew.float().numpy())
newplan={"candidate":"video_full_tag_rank6",
         "development_raw_pass":38,"development_safety_pass":39,
         "raw_splits":[f"tag_validation_{i}" for i in range(2,6)],
         "strip_splits":[f"tag_validation_{i}" for i in range(2,5)],
         "language_splits":[f"language_validation_{i}" for i in range(4)],
         "excluded_validation_passes":[36,37],
         "retain_rule":"positive tagged 95% paired CI; clean/language losses each <=2pp"}
(W/"revised_validation_plan.json").write_text(json.dumps(newplan,indent=2))
with (W/"notes.md").open("a") as f:
    f.write("\n### Pass37 frozen early-video gate, tagged validation1\n"+json.dumps(s37)+
            "\nPooled validation0+1: "+json.dumps(pool37)+
            "\nREJECT gate for futility; amend previous15-pass plan rather than spend remaining13 on "
            "an independently harmful edit. No gate retained. Validation0/1 excluded for replacement.\n"
            "New final candidate: video text-entry rank6 OBLIQUE full novel-prototype erasure "
            "P=I-A*QD^T, QD^T*A=I, QD orthogonal to protected clean/prefix/proprio span. "
            "Unlike pass33 orthogonal erasure, annihilates full category-mean novel context vectors. "
            "One tensor; no attention or hidden-neuron rewiring. Development38/39 then fresh "
            "tag validation2-5(512),strip2-4(384),language0-3(512).\n")
print("Edited tensors:",len(list(edited())),"relative weight change:",float((ew.float()-ow).norm()/ow.norm()))
```
