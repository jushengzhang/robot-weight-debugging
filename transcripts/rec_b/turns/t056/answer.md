Layer 14 remains abnormal after matching open commands: its attention contribution is about 4.79 versus 2.42 in the sampled controls. The gate-corrected direction is nearly orthogonal to the direction tested earlier, explaining why that test was weak. I will attenuate this corrected pathway by 50%, while protecting the other 31 readouts at the mean observed gate.
```python
for n in list(edited()):drop_edit(n)
l=14;beta=.5;u=gate_directions[l]
on=f"{E}{l}.self_attn.o_proj.weight"
O=load_original(on)
Oe=O-beta*torch.outer(u,u@O)
save_edit(on,Oe)
# Immediate, fixed-activation attribution of the edit; not an end-to-end model run.
raw=AT[:,l]
removed=beta*np.einsum("ntd,d->nt",raw,u.numpy())[:,:,None]*u.numpy()
post_removed=removed*ga[l-1][None]
grip_removed=post_removed@gn
other_removed=post_removed@Aout[other].T.numpy()
print("Saved gate-aware O14 rank1 attenuation beta",beta,
      "relative norm",float((Oe-O).norm()/O.norm()))
print("Fixed-activation gripper contribution removed F/S",
      np.round([grip_removed[~succ19].mean(),grip_removed[succ19].mean()],3).tolist())
print("Other-readout removed RMS per coordinate",
      np.round(np.sqrt((other_removed**2).mean((0,1)))[:6],4).tolist())
print("Protected direction count",len(other))
with (W/"notes.md").open("a") as f:
    f.write("\n## Gate-aware candidate G\n"
            "Residual identity reconstructed to0.14-1.32% relative RMS. Gates rotate raw gripper directions: "
            "cos(corrected14,raw)=.088. Open-command-matched gated L14 attention grip projection "
            "4.785 in failures vs2.415 in successes (effect1.18). "
            "Edit O14 <-(I-.5*u*u^T)O14, u=mean_gate14*g_read orthogonalized to mean_gate14*other31_readouts. "
            "No output/action scale changes. Prior raw-direction test D did not target this pathway.\n")
```
