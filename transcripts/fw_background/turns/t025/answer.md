The head-space covariances are strongly aligned even though the hidden coordinates differ. That supports treating shared head indices as meaningful. I will verify the algebra of the planned low-rank edit and prepare norm-matched control-head selection, without changing the checkpoint.

```python
def projected_head_block(block, basis, dose):
    u=torch.as_tensor(basis,dtype=block.dtype)
    if u.ndim==1:
        u=u[:,None]
    u=torch.linalg.qr(u,mode="reduced").Q
    return block-dose*u@(u.T@block)

# Algebraic tests on synthetic matrices only; this is not network execution.
gen=torch.Generator().manual_seed(713)
test_w=torch.randn(64,24*8,generator=gen)
test_u=torch.randn(64,2,generator=gen)
test_u=torch.linalg.qr(test_u,mode="reduced").Q
test_edited=test_w.clone()
test_slice=slice(5*8,6*8)
test_edited[:,test_slice]=projected_head_block(test_w[:,test_slice],test_u,0.4)
before=test_u.T@test_w[:,test_slice]
after=test_u.T@test_edited[:,test_slice]
keep=torch.ones(test_w.shape[1],dtype=torch.bool);keep[test_slice]=False
print("Synthetic projection tests:",
      {"unselected_columns_identical":bool(torch.equal(test_edited[:,keep],test_w[:,keep])),
       "dose_response_max_error":float((after-0.6*before).abs().max()),
       "finite":bool(torch.isfinite(test_edited).all())})

def matched_control_heads(layer, target_heads, excluded=()):
    ss=head_stats[layer*24:(layer+1)*24]
    blocked=set(target_heads)|set(excluded)
    chosen=[]
    for h in target_heads:
        t=ss[h]
        candidates=[j for j in range(24) if j not in blocked]
        def distance(j):
            c=ss[j]
            return sum(math.log(max(c[k],1e-9)/max(t[k],1e-9))**2
                       for k in ["o_norm","video_coupling","q_gain_rms","k_gain_rms"])
        j=min(candidates,key=distance)
        chosen.append({"target":h,"control":j,"relative_dose":t["o_norm"]/ss[j]["o_norm"],
                       "weight_geometry_distance":distance(j)})
        blocked.add(j)
    return chosen

print("Control utility demonstration (not an edit or selected hypothesis):",
      matched_control_heads(0,[8]))
print("Checkpoint edit count:",len(edited()))
del test_w,test_u,test_edited,before,after
```
