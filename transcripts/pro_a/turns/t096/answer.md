The final pass will replay four unchanged diagnostic observations and compare expert attention between the original model and the frozen edit, especially attention to base-camera patch 193. This tests the internal effect of the intervention; it cannot substitute for the unsuccessful rollout validation.

```forward
{"type":"probe","model":"both","inputs":["/mnt/SSD_16TB/jusheng/wsp_pi05/runs_pro/pro_a/work/roll_1_frames/e1946_seed7_original/t0000_inputs.pt","/mnt/SSD_16TB/jusheng/wsp_pi05/runs_pro/pro_a/work/roll_1_frames/e1923_seed7_original/t0000_inputs.pt","/mnt/SSD_16TB/jusheng/wsp_pi05/runs_pro/pro_a/work/roll_1_frames/e1562_seed7_original/t0000_inputs.pt","/mnt/SSD_16TB/jusheng/wsp_pi05/runs_pro/pro_a/work/roll_1_frames/e1490_seed7_original/t0000_inputs.pt"],"return":{"layers":[4,9,12,14],"what":"attn_maps"}}
```
