**Submitted: no weight edits.** `edited()` is empty, and the modified tensors were verified to match their originals. I did not find a validated robustness repair within the 40-pass budget.

The main evidence:
- **Late instruction-head amplification:** Layer 16 head 3 read meaningful instruction words, but increasing its contribution did not reduce wrong-task completions or improve success.
- **Selective late self-attention attenuation:** Replay probes showed mostly reduced movement amplitude, not improved instruction routing. On 640 fresh paired episodes, success fell from **379 to 369**, a **-1.56-point change (SE 1.13)**. The development position gain did not replicate.
- **Vision illumination correction:** The earlier promising candidate achieved only **+0.47 points (SE 0.95)** on a separate 640-episode validation set, insufficient evidence to retain it.

All candidates were therefore reverted rather than submitting an unsupported or harmful change. Unchanged weights preserve baseline performance, but **I do not claim any robustness improvement**.

The findings and final verification are recorded in `/work/notes.md` and `/work/final_report.json`.