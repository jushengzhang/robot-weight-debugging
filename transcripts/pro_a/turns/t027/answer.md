The early maps show that BOS dominates layers 0–2; substantive instruction attention first becomes appreciable around layer 4. I will also inspect the unusually large shared residual channels and their source weights. If one shared feature overwhelms lexical or spatial detail, a channel-specific repair may be more appropriate than amplifying entire heads.
```python
embed_name=f"{vlm_root}.embed_tokens.weight"
emb=load_original(embed_name)
print("Embedding rows: norm, dominant dimensions (before sqrt(width) scaling)")
for text in ["<bos>","bbq","tomato","milk","bowl","plate","pick","\n"]:
    ti=sp.bos_id() if text=="<bos>" else sp.encode(text)[-1]
    v=emb[ti]
    tops=v.abs().topk(3).indices.tolist()
    print(repr(text),ti,round(float(v.norm()),3),
          [(d,round(float(v[d]),3)) for d in tops],
          "d674/738",round(float(v[674]),4),round(float(v[738]),4))
proj_bias_n="paligemma_with_expert.paligemma.model.multi_modal_projector.linear.bias"
pb=load_original(proj_bias_n)
print("Projector bias",round(float(pb.norm()),2),
      [(int(d),round(float(pb[d]),3)) for d in pb.abs().topk(6).indices])
print("VLM outlier weights: RMSNorm weights, v_proj column norms, MLP row norm / median")
for l in [0,1,3,6,9,12,14,15,16]:
    nw=load_original(f"{vlm_root}.layers.{l}.input_layernorm.weight")
    vv=load_original(f"{vlm_root}.layers.{l}.self_attn.v_proj.weight")
    dd=load_original(f"{vlm_root}.layers.{l}.mlp.down_proj.weight")
    rn=dd.norm(dim=1)
    print(l,"norm(w) median",round(float(nw.median()),3),
          "norm(w) [674,50,738]",rounded(nw[[674,50,738]],3),
          "vcols",rounded(vv[:,[674,50,738]].norm(dim=0),3),
          "down_row_ratios",rounded(rn[[674,50,738]]/rn.median(),2))
del emb,dd,vv
# Candidate with actual content-reading heads, distinct from the failed newline head.
clear_edits()
set_head_gains([(4,3,1.5),(4,7,1.5)])
snapshot("candidate_L4H37_o150")
print("Current candidate:",edited())
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 9: early attention maps\n"
            "Layers0-2 dominated by BOS; first appreciable content attention at L4: "
            "H3 words25-32%, H7 words26-35%, versus ~0 at L0. "
            "Initial task/semantic image differences are nonzero for all matched groups, so no pure prompt contrast is available.\n"
            "Candidate D: only L4H3/H7 output columns x1.5, testing early content reading vs failed late newline amplification.\n")
```
