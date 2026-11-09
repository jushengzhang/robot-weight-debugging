The saved probes contain attention-output summaries for all ten denoising calls. I will use the known value-write directions to estimate each head's feature and reference contribution over time, checking reconstruction error so that unreliable decompositions are not treated as measurements.
```python
def attention_deltas(k):
    rr = json.loads((W/f"probe_{k}"/"results.json").read_text())
    by = {(r["arm"],r["idx"]):r for r in rr}
    n = max(r["idx"] for r in rr)+1
    key = "calls::expert.17.attn_out::action"
    return torch.from_numpy(np.stack([
        np.load(by[("edit",i)]["activations"])[key] -
        np.load(by[("orig",i)]["activations"])[key] for i in range(n)
    ])).float()

R = designs["clearance_attention_scores.pt"]["routing_matrix"].float()
Rp = torch.linalg.pinv(R)
D41, D42 = attention_deltas(41), attention_deltas(42)
S41, S42 = D41 @ Rp.T, D42 @ Rp.T
for label,D,S in [("feature",D41,S41),("reference",D42,S42)]:
    errors = ((D-S@R.T).square().sum((0,2))/D.square().sum((0,2))).sqrt()
    print(label, "reconstruction error by denoising call",
          [round(float(x),4) for x in errors])
ctrl = cd["source_control_indices"]
for group in ["source_empty","source_contact","check_empty","check_contact","control_open_control"]:
    inds = [i for i,r in enumerate(m) if group_key(r)==group]
    print("\n"+group)
    for h in [1,7]:
        centered = S41[inds,:,h] - cd["betas"][h]*S42[inds,:,h]
        print("head",h,"centered by call",
              [round(float(x),3) for x in centered.mean(0)])
print("\nsource-control reference means by head/call:")
for h in [1,7]:
    print(h, [round(float(x),3) for x in S42[ctrl,:,h].mean(0)])
print("parts reproduce original value direction:",
      float((sum(parts.values())-designs["clearance_centering_design.pt"]["value_direction"]).norm()))
```
