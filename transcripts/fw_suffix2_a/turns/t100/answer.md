I’ll compare the probe actions with the original reference, separating clean drift, paraphrase drift, and the remaining raw-to-clean gap. These measurements can flag problems, but rollout success will still decide whether the edit is useful.
```python
m28 = probe_metrics(28)
print("Largest clean drifts:")
base = np.stack([np.load(W/"probe_3"/f"probe_orig_{i}_action.npy") for i in range(56)])
cur = np.stack([np.load(W/"probe_28"/f"probe_edit_{i}_action.npy") for i in range(56)])
rms_each = np.sqrt(((cur-base)**2).mean(axis=(1,2)))
for i in sorted(range(24,56), key=lambda i:-rms_each[i])[:6]:
    print(i, pitems[i]["category"], round(float(rms_each[i]),6),
          pitems[i]["suite"], pitems[i]["id"])
print("Mean raw/clean/language RMS:", [round(float(rms_each[a:b].mean()),6) for a,b in [(0,24),(24,48),(48,56)]])
```
