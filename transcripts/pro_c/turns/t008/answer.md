I will summarize each head’s attention to base-camera, wrist-camera, and real instruction tokens, including concentration within the image grid. This distinguishes a language-routing bottleneck from heads that repeatedly attend to fixed scene locations.
```python
p2=json.loads((W/"probe_2/results.json").read_text())
print("probe structure:",str(p2)[:5000])
```
