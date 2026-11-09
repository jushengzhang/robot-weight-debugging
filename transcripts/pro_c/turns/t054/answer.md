Layer 16 has a large visual-common MLP direction: over 92% of its camera-mean magnitude remains after projecting out the measured language-output span. I will halve only that rank-one output component. This tests whether the final KV cache is burdened by a large, mostly instruction-independent visual offset; it does not remove whole MLPs or rescale actions.
```python
l=16
lang=mat(a17,f"mean::vlm.{l}.mlp_out::lang")
vis=torch.cat([mat(a17,f"mean::vlm.{l}.mlp_out::base"),
               mat(a17,f"mean::vlm.{l}.mlp_out::wrist")])
q=ortho_columns(torch.cat([lang.T,torch.eye(2048)[:,[50,604,674,738]]],dim=1))
mu=vis.mean(0)
u=mu-q@(q.T@mu)
u=u/u.norm()
visual_offset={"layer":16,"u":u,"protected_rank":q.shape[1]}
torch.save(visual_offset,W/"visual_offset_definition.pt")
n=f"{VLM}.layers.16.mlp.down_proj.weight"
orig=load_original(n)
alpha=0.5
save_edit(n,orig-alpha*torch.outer(u,u@orig))
print("visual direction fraction",round(float((vis@u).norm()/vis.norm()),3),
      "language leakage",float((lang@u).norm()/lang.norm()),
      "relative weight edit",round(float((load(n)-orig).norm()/orig.norm()),4))
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass17 diagnosis and candidate F\nMLP probes32 matched initial semantic/task inputs. VLM16 MLP camera mean norm base4604/wrist2909 vs lang370; camera variation >99% rank4. >92% of common visual output is orthogonal to span of32 language means. Candidate F: halve rank1 visual-common output direction of VLM16 MLP down_proj, after protecting all32 language means and sink coordinates. Only one tensor; purpose: reduce visual common offset injected immediately before final prefix KV, not claimed to erase scene identity. No other edits.\n")
print("edited",edited())
```
