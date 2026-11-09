The milder query edit also lost one net success and showed no consistent benefit, so I will revert it. The residual analysis suggests gripper decisions form earlier than the late heads we tested. I will isolate action-plan recycling more directly: remove one gripper-plan direction from layer 7’s **expert value projection**, which changes action-token feedback but leaves all visual/language prefix values intact.
```python
for n in list(edited()): drop_edit(n)
# Recover layer-7 normalization coefficients from the already measured time condition.
L=7
nw=load_original(f"{expert}layers.{L}.input_layernorm.dense.weight")
nb=load_original(f"{expert}layers.{L}.input_layernorm.dense.bias")
mod7=cond@nw.T+nb
rr=torch.from_numpy(r15[:,ls.index(6),:,:]).float()
nr=rr/(rr.square().mean(-1,keepdim=True)+1e-6).sqrt()
nx=nr*(1+mod7[None,:,:1024])+mod7[None,:,1024:2048]
gripavg=np.asarray(acts15)
print("Output grip class counts by initial-noise group:")
for g in np.unique(gp):
    print(int(g),"open",int(np.sum((gp==g)&(gripavg<-.6))),
          "close",int(np.sum((gp==g)&(gripavg>.6))))
dirs=[]
for call in range(3):
    parts=[]
    for g in np.unique(gp):
        pos=(gp==g)&(gripavg>.6); neg=(gp==g)&(gripavg<-.6)
        if pos.sum()>=2 and neg.sum()>=2:
            parts.append(nx[pos,call].mean(0)-nx[neg,call].mean(0))
    dirs.append(torch.stack(parts).mean(0))
dc=torch.stack(dirs)
print("Early-call contrast cosine",np.round(torch.nn.functional.normalize(dc,dim=1).numpy()@torch.nn.functional.normalize(dc,dim=1).numpy().T,3).tolist())
d=dc.mean(0)
ain=load_original("action_in_proj.weight")
# Protect shared activation and direct pose-encoding directions.
protection=[nx[:,:3].mean((0,1))]
for j in range(6):
    protection.append(ain[:,j]*(1+mod7[:3,:1024]).mean(0))
P,_=torch.linalg.qr(torch.stack(protection,1),mode="reduced")
d=d-P@(P.T@d)
d=d/d.norm()
vn=f"{expert}layers.7.self_attn.v_proj.weight"
vv=load_original(vn)
dv=-.5*torch.outer(vv@d,d)
save_edit(vn,vv+dv)
torch.save({"direction":d,"layer":7,"calls":[0,1,2],"alpha":.5},W/"expert_self_grip_direction_L7.pt")
(W/"current_candidate.json").write_text(json.dumps({"name":"E7_V_self_grip_attenuate_0.5","alpha":.5,"layer":7,"direction_file":"expert_self_grip_direction_L7.pt"}))
print("Edit fraction",float(dv.norm()/vv.norm()),"direction scores std",float((nx@d).std()))
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 33: E16H3 query gain 0.75 rejected\n"
            "- 31/64 -> 30/64, -1.6 pp (SE 3.5), 2 rescues / 3 losses; loops unchanged. No consistent improvement across query doses. Reverted.\n"
            "- New structural test: L7 expert v_proj attenuates one existing gripper-plan input direction, derived from within-noise-group open/close contrasts. VLM prefix values remain unedited.\n"
            "- Mean activation and direct pose-encoding directions protected. This isolates action self-feedback rather than weakening visual feedback or an entire mixed head.\n")
```
