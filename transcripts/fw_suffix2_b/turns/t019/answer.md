Before collecting more slices, I’ll verify that the readouts are stable across diffusion steps. If the isolated coordinates vary with the action state, they would not be reliable measurements of the fixed text context.```python
readout_dir = W / "context_readouts"
readout_dir.mkdir(exist_ok=True)
def decode_readout(k, start):
    recs = probe_records(k)
    vals, time_err = [], []
    for rec in recs:
        z = np.load(rec["activations"])
        per_layer = np.stack([z[f"calls::action.{i}.resid::action"][:,:read_dim]
                              for i in range(30)], axis=0)
        # [layer, diffusion call, coordinate]
        dif = np.diff(np.concatenate([np.zeros_like(per_layer[:1]), per_layer], axis=0), axis=0)
        dif[1::2] *= -1
        vals.append(dif.mean(axis=1))
        time_err.append(rms(dif-dif.mean(axis=1,keepdims=True)))
    out = np.stack(vals)
    np.save(readout_dir/f"slice_{start:04d}.npy",out)
    print("Pass",k,"slice",start,"shape",out.shape,"RMS",rms(out),
          "max step drift",max(time_err),"median step drift",float(np.median(time_err)))
    for j in [0,1,2,5,15,16,20,29]:
        print("query",j,"raw norm",round(float(np.linalg.norm(out[:24,j],axis=1).mean()),4),
              "clean norm",round(float(np.linalg.norm(out[24:48,j],axis=1).mean()),4),
              "paired contrast norm",round(float(np.linalg.norm(out[:24,j]-out[24:48,j],axis=1).mean()),4))
    return out
sl0 = decode_readout(7,0)
set_readout_slice(480)
print("Next readout slice480:960 installed.")
with (W/"notes.md").open("a") as f:
    f.write("\nPass7: first480-coordinate,30-query context readout acquired; decoded in context_readouts/. "
            "Instrumentation proceeds to slice480:960.\n")
```
