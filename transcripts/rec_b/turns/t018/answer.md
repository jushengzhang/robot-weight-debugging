I will inspect the returned residual structure and compare camera-specific feature norms with the gripper-related action residuals. I will also check how strongly the learned action-input gripper direction enters each expert query head, which can reveal a feedback route from a tentative grasp command back into visual selection.
```python
p7=json.loads((W/"probe_7/results.json").read_text())
print("probe residual example:",str(p7[0])[:2800])
actpath=p7[0].get("activations")
ar7=dict(np.load(actpath)) if actpath else {}
print("activation shapes:",[(k,v.shape) for k,v in ar7.items()][:55])
ain=load_original("action_in_proj.weight")
print("action input column norms",ain.norm(dim=0).numpy().round(3).tolist())
gvec=ain[:,6]/ain[:,6].norm()
for L in range(18):
    q=load_original(f"{EX}.layers.{L}.self_attn.q_proj.weight").reshape(8,256,1024)
    # Average amplification of this fixed input direction relative to isotropic directions.
    ratios=(torch.einsum("hkd,d->hk",q,gvec).norm(dim=1)/(q.flatten(1).norm(dim=1)/32))
    print("L",L,"Q gripper-direction ratios",ratios.numpy().round(2).tolist())
print("ada norm bias statistics (scale/shift/gate chunks, order to be confirmed by effects):")
for L in [0,3,5,7,11,14,17]:
    b=load_original(f"{EX}.layers.{L}.input_layernorm.dense.bias").reshape(3,1024)
    print(L,[(round(float(t.mean()),3),round(float(t.std()),3),round(float(t.abs().max()),3)) for t in b])
```
