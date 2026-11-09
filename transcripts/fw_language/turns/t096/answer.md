A query bias can make a text-attention head prefer certain tokens regardless of the observation. I will screen for unusually strong query biases and attention temperatures, concentrating on the layers where the fixed-observation instruction contrasts first grew.
```python
action_screen = [x for x in temperature_screen if x["expert"] == "action"]
video_screen = [x for x in temperature_screen if x["expert"] == "video"]
for label, rows in [("action", action_screen), ("video", video_screen)]:
    print(label, "largest query-bias ratios:")
    for x in sorted(rows, key=lambda x: x["q_bias_ratio"], reverse=True)[:8]:
        print(x)
    print(label, "largest within-layer temperature ratios:")
    for x in sorted(rows, key=lambda x: x["relative"], reverse=True)[:8]:
        print(x)
print("Action L10-13 largest query-bias ratios:")
for x in sorted([x for x in action_screen if 10 <= x["layer"] <= 13],
                key=lambda x: x["q_bias_ratio"], reverse=True)[:8]:
    print(x)

# Recover the exact original probe input ordering, without altering any input.
p4_results = json_structure(work / "probe_4" / "results.json", 1500)
print("Current edits:", list(edited()))
```
