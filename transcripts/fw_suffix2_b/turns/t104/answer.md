The clean estimate is now +1.3 points over 384 episodes, while the tagged estimate is +3.4 points. I’ll save the updated status and prepare the third paraphrase batch. The final decision will use all planned validation data, not these interim estimates.
```python
status = validation_status()
notes = (W/"notes.md").read_text()
notes = notes.replace(
    "## Current state\nNo edits. 0/50 forward passes used.",
    "## Current state\nFROZEN validation candidate since pass31: 308 tensors, "
    "rank256 oblique text-entry repair plus single suffix q/k gate in BOTH experts. "
    "Exact snapshot: candidates/hybrid_r256_d1_t05_g128; frozen_manifest.json hashes. "
    "No further tuning. validation_plan.json contains ALL requests through pass50. "
    "Load analysis_helpers.py then validation_helpers.py to summarize results and print "
    "the next request. Latest status is in validation_status.json and chronological notes below. "
    "Primary target:1024 raw; safety:768 strip and640 language, all paired and independent "
    "of edit design. Retain only if tagged gain clearly exceeds SE and safety losses are "
    "no worse than about2pp; otherwise remove all edits."
)
(W/"notes.md").write_text(notes)
```
