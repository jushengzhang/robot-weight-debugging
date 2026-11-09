This panel adds a small positive difference, but uncertainty remains substantial. I’ll update the cumulative paired and clustered estimates, then load the next fixed panel without changing the edit.
```python
def log_frozen_rollout(k, purpose):
    result=paired_summary(paired_rows([k]))
    ledger=json.loads((root/"pass_ledger.json").read_text())
    if not any(e.get("pass")==k for e in ledger):
        ledger.append({"pass":k,"type":"rollout","edit":"readout_camera_rank2_half",
                       "purpose":purpose,**result})
        (root/"pass_ledger.json").write_text(json.dumps(ledger,indent=2))
        with (root/"notes.md").open("a") as f:
            f.write(f"\n## Pass{k}: {purpose}\n- Frozen candidate unchanged. {result}\n")
    return result

print("Pass29:",log_frozen_rollout(29,"Preplanned camera seed panel"))
current_camera_passes=[20,21,22,23,28,29]
print("Cumulative independent camera:",clustered_summary(paired_rows(current_camera_passes)))
assert hashlib.sha256(load(wname).contiguous().numpy().tobytes()).hexdigest()==plan["tensor_sha256"]
print("NEXT REQUEST:",json.dumps(json.loads((root/"request_30.json").read_text()),separators=(",",":")))
```
