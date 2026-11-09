Layer 14 heads 0, 1, and 3 show the strongest instruction-dependent visual redirection; their average visual-attention changes are substantially larger than neighboring heads. I will increase only these three output blocks by 50%, with the original embeddings restored. This tests whether strengthening the actual language-conditioned visual route helps new goals override scene habits.
```python
assert not edited()
ground_heads=[0,1,3]
n=f"{ex}.layers.14.self_attn.o_proj.weight"
a=load_original(n)
for h in ground_heads:
    a[:,h*256:(h+1)*256] *= 1.5
save_edit(n,a)
dst=W/"candidate_grounding_heads_g15"
dst.mkdir(exist_ok=True)
for p in (W/"edited").iterdir():
    if p.is_file(): shutil.copy2(p,dst/p.name)
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 21 and candidate grounded-vision route gain\n"
            "Controlled noun swap: expertL14 H0/H1/H3 have mean visual attention TV "
            ".269/.209/.220; neighboring strongest L13H3=.125. "
            "H0 andH1 are >90% visual-attending and redirect visual peaks under "
            "pure language intervention. Original embeddings restored. "
            "Candidate: expertL14 o_proj column blocks H0,H1,H3 x1.5; "
            "only one tensor edited. Prediction: strengthen instruction-conditioned "
            "visual control, unlike earlier direct-word-head gain.\n")
(W/"current_candidate.json").write_text(json.dumps({"name":"grounding_heads_g15","layer":14,"heads":ground_heads,"gain":1.5}))
print("Active edit:",edited(),"relative weight delta",
      round(float((a-load_original(n)).norm()/load_original(n).norm()),4))
```
