Halving the wrist head produced only a small, statistically unclear gain and one regression, so I will discard it. The next test is more selective: reduce only the gripper-input component of layer 2 head 0’s queries, while preserving its response to every other action-input direction. This targets a possible self-reinforcing grasp hypothesis rather than suppressing wrist perception.
```python
reset_edits()
# Orthogonalize the gripper input direction against all other action-input columns.
# This is a fixed analytic projection, not an optimization or fit to success labels.
other_cols=[j for j in range(32) if j!=6]
qb,_=torch.linalg.qr(ain[:,other_cols],mode="reduced")
g_specific=ain[:,6] - qb@(qb.T@ain[:,6])
g_specific=g_specific/g_specific.norm()
print("gripper-specific direction overlap with raw gripper",float(g_specific@gvec))
print("max overlap with other action columns",float((g_specific@ain_u[:,other_cols]).abs().max()))
def edit_query_direction(spec, direction):
    # spec {layer: {head: removed_fraction}}
    for L,hm in spec.items():
        n=f"{EX}.layers.{L}.self_attn.q_proj.weight"
        t=load_original(n).clone()
        for h,beta in hm.items():
            sl=t[h*256:(h+1)*256]
            sl -= beta*torch.outer(sl@direction,direction)
        save_edit(n,t)
edit_query_direction({2:{0:0.5}},g_specific)
n=f"{EX}.layers.2.self_attn.q_proj.weight"
orig=load_original(n); change=load(n)-orig
print("relative tensor change",round(float(change.norm()/orig.norm()),5),
      "fraction changed after bf16",round(float((load(n).to(torch.bfloat16)!=orig.to(torch.bfloat16)).float().mean()),3))
print("raw grip query norm before/after",
      float((orig[:256]@g_specific).norm()),float((load(n)[:256]@g_specific).norm()))
with (W/"notes.md").open("a") as f:
    f.write("\n## Rejected candidate B, pass 8\nL5h3 O x0.5:26/48 ->27/48, +2.1 +/-3.6 pp; "
            "2 rescues/1 regression, loops21->19. No persuasive gain; reset.\n"
            "\n## Candidate C: gripper-feedback direction\nAction input column6 orthogonalized against other31 "
            "columns gives a gripper-specific residual direction. Expert L2h0 Q amplifies raw gripper4.14x "
            "versus~1x for physical action columns0..5; head has~60% self-attention in inspected maps. "
            "Apply Q_h <- Q_h (I - beta*g*g^T), beta=.5, only L2h0. "
            "No change to action output or global action/noise scale; all other input-column directions preserved.\n")
```
