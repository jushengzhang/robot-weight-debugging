Rank 320 captures 99.4% of the measured tag-contrast energy, versus 97.7% at rank 256, without the sharp amplification seen at higher ranks. I’ll test that one stronger projection on the same development episodes, leaving the suffix gate unchanged.
```python
cfg = update_hybrid_projection(rank=320, dose=1.0)
cfg["basis_file"] = "rich_context_projection.pt"
(W / "current_candidate.json").write_text(json.dumps(cfg, indent=2))
snapshot_candidate("hybrid_r320_d1_t05_g128")
with (W / "notes.md").open("a") as f:
    f.write("\nPass30 planned: broad hybrid rank320 dose1; same gate. "
            "Rank320 captures99.39% contrast energy vs97.71% for256; "
            "oblique RMS amplification1.986 vs1.765. "
            "One stronger spectral-dose test on development96; "
            "selection by rollouts, not diagnostic action distances.\n")
req = {"type":"rollout", "model":"both", "language":"raw",
       "episodes":[{k:r[k] for k in ("suite","id","seed")} for r in sp["development"]]}
(W / "request_development_raw.json").write_text(json.dumps(req))
print("EDIT COUNT", len(edited()), "CONFIG", cfg)
print(json.dumps(req, separators=(",",":")))
```
