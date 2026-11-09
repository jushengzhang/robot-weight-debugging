The protected projection behaved exactly as intended internally—the targeted component retained 50.04% of its amplitude—but changed actions by only about 0.0006 RMS and did not improve agreement between instruction pairs. I will revert it. To finish testing the earlier head-14 hypothesis, I will separate the video and action branches; the combined intervention could have hidden opposite effects.
```python
print(set_head_output_edit(["video"],13,14,0.5,"L13H14_video_half"))
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 9 causal localization, negative behavioral result\n"
            "- Protected action L11 self-attention rank-one half-projection attenuated the measured component to 0.5004 of original, confirming attn_out records the self-attention output.\n"
            "- Mean action RMS shifts only .00061 for both successful and failed prompts; paired action gap .06923 -> .06931 (no improvement). REVERTED. Removing error directions orthogonal to the task span has too little behavioral leverage here.\n"
            "- Next predefined branch-discrimination tests: L13 H14 video-only and action-only output attenuation. Combined pass-7 edit may mask opposing effects; no success claimed.\n")
```
