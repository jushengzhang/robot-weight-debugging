I will check whether the source-feature edit produces a selective behavioral effect and whether the successful recovery query remains stable. This also tests the redundancy explanation for why single-head attenuations had so little effect.

```python
pr16=json.loads((W/"probe_16/results.json").read_text())
s16=paired_probe_summary(pr16,{2:[7],8:[2,7],12:[2,3],16:[3]})
# Inspect bf16 survival rather than assuming a rank-one float32 edit survives casting.
for n in edited():
    a=load_original(n);b=load(n)
    neq=int(torch.count_nonzero(a.to(torch.bfloat16)!=b.to(torch.bfloat16)))
    print("bf16-different entries",n.split(".")[-2:],neq,"/",a.numel())
```
