The full prototype projection failed decisively, losing 20 of 96 successes, so I’ll remove it immediately. For the remaining tests I’ll use the earlier, much smaller orthogonal rank-six edit. Its one-success development gain is weak; the independent paired tests will determine whether it deserves to be submitted at all.
```python
s38=paired_rollout(38)
for n in list(edited()):
    drop_edit(n)
name=V+"text_embedding.0.weight"
save_edit(name,torch.from_numpy(np.load(W/"candidate_video_tag_rank6.npy")))
final_rank6=W/"frozen_video_orthogonal_rank6"
shutil.copytree(W/"edited",final_rank6,dirs_exist_ok=True)
shutil.copy2(W/"novel_tag_rank6_projection.pt",final_rank6/"metadata.pt")
rank6_manifest=tensor_manifest(final_rank6)
(W/"frozen_video_orthogonal_rank6_manifest.json").write_text(json.dumps(rank6_manifest))
newplan["candidate"]="frozen_video_orthogonal_rank6"
newplan["development_raw_pass"]=33
newplan["development_safety_pass"]=39
(W/"revised_validation_plan.json").write_text(json.dumps(newplan,indent=2))
with (W/"notes.md").open("a") as f:
    f.write("\n### Pass38 full novel-prototype oblique projection REJECTED\n"+json.dumps(s38)+
            "\n20 losses,0 gains: full prototype components contain indispensable information. "
            "Removed immediately; no safety rollout warranted.\n"
            "FINAL TEST CANDIDATE: frozen_video_orthogonal_rank6, one video text-entry tensor "
            "(pass33 development47->48,3 gains/2 losses). Original minus W*Q6*Q6^T; "
            "relative weight norm change.0391. No gating, no other edits. "
            "Use pass39 development safety,40-43 fresh tag validation2-5,44-46 strip2-4,"
            "47-50 language0-3. No further candidate search. Retain only positive tagged95%CI "
            "and clean/language losses each<=2pp; otherwise original.\n")
print("Rejected full projection:",s38)
print("Final candidate edits:",list(edited()))
print("Frozen hash:",rank6_manifest)
```
