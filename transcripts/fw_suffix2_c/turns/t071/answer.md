I’ll compare the two video ranks and then move to paired simulator rollouts. The action probes are useful diagnostics, but task success and clean-language safety must determine the final choice.
```python
m24=compare_screen(24)
splits=json.loads((W/"splits.json").read_text())
print("SPLIT SIZES", {k:len(v) for k,v in splits.items()})
# Prepare exact paired-development request, not a new sample.
devreq={"type":"rollout","model":"both","language":"raw",
        "episodes":[{k:e[k] for k in ["suite","id","seed"]} for e in splits["development"]]}
(W/"request_dev_both.json").write_text(json.dumps(devreq))
print("DEVELOPMENT REQUEST",json.dumps(devreq))
```
