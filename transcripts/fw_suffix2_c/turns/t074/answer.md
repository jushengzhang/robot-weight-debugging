The development rollout is not convincing: tagged success rises by only one episode, while paraphrase success falls by two. I’ll make one final diagnostic distinction before validation: mask the detected suffix tokens while taking weighted summaries of the remaining context. This tests prefix-embedding contamination rather than simply rediscovering the suffix itself.
```python
print("ROLLOUT RECORD SAMPLE")
rows25=[json.loads(x) for x in (W/"roll_25.jsonl").read_text().splitlines() if x.strip()]
print(str(rows25[0])[:1800])
print("OLD DETECTOR NORM/BIAS",
      float(u_tag.norm()),
      float(np.load(W/"fwd/probe_10_weights/mot.mixtures.action.text_embedding.0.bias.npy")[-1]))
# Save a separate spec and separate script so earlier readouts remain reproducible.
masked_spec=dict(spec)
masked_spec["m"]=1016
excluded24=torch.argsort(v2.square().sum(0))[:24]
masked_spec["feature_ids"]=torch.tensor([i for i in range(3072) if i not in set(excluded24.tolist())])
masked_spec["excluded"]=excluded24
masked_spec["detector_weight"]=u_tag
masked_spec["detector_bias"]=float(np.load(W/"fwd/probe_10_weights/mot.mixtures.action.text_embedding.0.bias.npy")[-1])
torch.save(masked_spec,W/"masked_readout_spec.pt")

masked_code=instrument_code.replace("weighted_readout_spec.pt","masked_readout_spec.pt")
masked_code=masked_code.replace("install_weighted_readout","install_masked_readout").replace("collect_weighted_readout","collect_masked_readout")
masked_code=masked_code.replace('W/f"weighted_{active[\'expert\']}_{active[\'slice\']}.npy"',
                                'W/f"masked_{active[\'expert\']}_{active[\'slice\']}.npy"')
masked_code=masked_code.replace('    save_edit(p+"text_embedding.0.weight",w)',
'''    w[m+6]=spec["detector_weight"].float()
    b[m+6]=spec["detector_bias"]
    save_edit(p+"text_embedding.0.weight",w)''')
masked_code=masked_code.replace('    save_edit(p+"text_embedding.2.weight",z)',
'''    z[m+3,m+6]=1./32.
    save_edit(p+"text_embedding.2.weight",z)''')
masked_code=masked_code.replace('a[anchors]=gain if direction>=0 else 1.',
                                 'a[anchors]=10000.')
oldkey='''        if direction>=0:
            a[anchors,m+direction]=1.
        else:
            a[-1]=load_original(cp+"k.weight")[-1]'''
newkey='''        a[anchors,m+3]=-128.
        if direction>=0:
            a[anchors,m+direction]=gain'''
assert oldkey in masked_code
masked_code=masked_code.replace(oldkey,newkey)
masked_code=masked_code.replace('a=torch.ones_like(load_original(cp+"k.bias")); a[anchors]=0;',
                                 'a=torch.ones_like(load_original(cp+"k.bias"))*10000.; a[anchors]=0;')
(W/"masked_readout.py").write_text(masked_code)
exec(masked_code)
install_masked_readout(0)
with (W/"notes.md").open("a") as f:
    f.write("\nPass25 development result for video1/action.5: tagged49->50/96 (+1.04pp); language26->24/32 (-6.25pp, low power). Not sufficient to retain. FINAL diagnostic campaign: masked weighted prefix summaries, using same safe threshold.5 suffix detector to suppress detected suffix tokens in all13 queries. 3 slices1016 video features; masked_readout.py and masked_readout_spec.pt. Temporary instrumentation active, must clear after passes26-28. Plan repair probe29/development30 then freeze and independent validation.\n")
```
