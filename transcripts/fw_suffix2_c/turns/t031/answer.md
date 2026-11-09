I will measure the rank-one result, then test a higher-rank version that also removes secondary tag-contrast directions. Comparing the two gives a causal dose-response while changing only the same video text-projection tensor.
```python
m11=screen_metrics(11)
shutil.copyfile(W/"current_projector.pt",W/"projector_P1.pt")
save_dir=W/"candidate_video_hidden_P1"; save_dir.mkdir(exist_ok=True)
for n in edited():
    np.save(save_dir/(n+".npy"),load(n).numpy())
apply_hidden_projector(8,1.0,"uniform")
```
