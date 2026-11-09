The control check passed: paired inputs differ only in the prompt and the saved action record, not in images, proprioception, or seed. I will now add the same gate to the video expert to test the indirect route from text-conditioned video features into action self-attention.
```python
print(json.dumps(apply_attention_gate(("action","video"),6.,threshold=.5),indent=2))
with (W/"notes.md").open("a") as f:
    f.write("\n### Input control verified\n"
            "All 24 raw/clean initial pairs have exactly identical image tensors, proprio and seeds; "
            "only prompt and saved action_chunk differ. Both-expert gate strength6 threshold.5 now applied "
            "to test the video-mediated route in addition to the action-only gate.\n")
```
