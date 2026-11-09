Adding the detector-dependent key/value signal recovered a development gain: 47 to 51 successes, with five gains and one regression. I’ll save this candidate and run the same clean/paraphrase safety check. It is still a development result, not independent evidence of improvement.
```python
s30 = paired_rollout(30)
snapshot = W/"candidate_null_payload1"
shutil.copytree(W/"edited", snapshot, dirs_exist_ok=True)
torch.save({"carrier":carrier,"payload":payload,"hidden":hidden,
            "threshold":.20,"gain":128.,"dose":1.,
            "development_pass":30}, snapshot/"metadata.pt")
with (W/"notes.md").open("a") as f:
    f.write("\n### Pass30 null-carrier gate plus legacy K/V payload\n"+json.dumps(s30)+
            "\nSaved immutable candidate_null_payload1. Same detector-off computation as isolated gate; "
            "retains original text output channel. Next safety repeats pass27's80 strip+48 language.\n")
print("Pass30:", s30)
print("Snapshot tensor files:",len(list(snapshot.glob("*.npy"))))
```
