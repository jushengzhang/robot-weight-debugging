The isolated gate produced no net development gain. One difference from the earlier candidate is that the earlier detector also entered ordinary key/value projections through its carrier channel. I’ll test that route explicitly: add back the same detector-dependent signal while retaining the original clean channel and the isolated gate readout.
```python
s29 = paired_rollout(29)
channel = int(meta["specs"]["video"]["channel"])
payload = torch.zeros_like(carrier)
payload[channel] = 1.
payload -= carrier * carrier[channel]
tw = load_original(V+"text_embedding.2.weight").clone()
tw[:,hidden] = (carrier+payload)[:,None]/128.
save_edit(V+"text_embedding.2.weight",tw)
torch.save({"carrier":carrier,"payload":payload,"hidden":hidden,
            "threshold":.20,"gain":128.,"dose":1.},
           W/"null_carrier_payload_metadata.pt")
with (W/"notes.md").open("a") as f:
    f.write("\n### Pass29 isolated null-carrier gate development\n"+json.dumps(s29)+
            "\nNo net gain. Next controlled test adds the legacy detector-dependent K/V payload "
            "in channel1066, orthogonalized against the null carrier. Original clean channel retained; "
            "detector-off computation unchanged from pass28/29 apart from roundoff.\n")
print("Pass29:",s29)
print("payload norm/carrier overlap:",float(payload.norm()),float(payload@carrier))
print("Active edits:",len(list(edited())))
```
