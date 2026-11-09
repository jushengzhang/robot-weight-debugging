Suppressing the late self-focused heads hurts recovery (-5.6 points, 4.0-point paired SE), so I will remove that edit. The next test preserves a visual head's weight norm but changes its relative-position preference: a one-camera-offset rotary rotation of layer 8 head 2's query rows. The hypothesis is that this wrist specialist can be redirected toward base-camera evidence when the target is no longer near the gripper.

```python
p10=paired_roll(10)
def set_query_rope(l,h,delta,blend=1.0,theta=10000.0,clear=True):
    if clear: clear_edits()
    n=f"{E}.layers.{l}.self_attn.q_proj.weight"
    w=load_original(n)
    q=w[h*256:(h+1)*256].clone()
    freq=theta**(-torch.arange(0,256,2,dtype=torch.float32)/256)
    angle=float(delta)*freq
    c=angle.cos()[:,None];s=angle.sin()[:,None]
    rot=torch.cat([q[:128]*c-q[128:]*s,q[128:]*c+q[:128]*s],dim=0)
    w[h*256:(h+1)*256]=(1-blend)*q+blend*rot
    save_edit(n,w)
    return n, q, w[h*256:(h+1)*256]
n,q0,q1=set_query_rope(8,2,-256,1.0)
print("candidate",n,"head2 rotary delta=-256 (standard Gemma theta=10000)")
print("head norm before/after",float(q0.norm()),float(q1.norm()),
      "relative change",float((q1-q0).norm()/q0.norm()))
checkpoint_edits("candidate_L8H2_rope_m256","Diagnostic: norm-preserving -256-position RoPE rotation of L8H2 query rows; test wrist->base retargeting.")
with open(W/"notes.md","a") as f:
    f.write("\n## Passes9-10: late self-head attenuation\nL17 H0/H3/H4/H6 factor0.5 caused appreciable action changes but reduced success63/126 vs70/126 (-5.6+/-4.0pp SE), 9 wins/16 losses, loops49 vs46. REJECTED and removed.\nNext: norm-preserving rotary rotation of expert L8H2 q_proj by -256 token positions (standard Gemma RoPE theta10000), aiming to shift wrist-specialist relative-position preference by one camera segment. Probe must confirm actual redistribution before interpreting.\n")
```
