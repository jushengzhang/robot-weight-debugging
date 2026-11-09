I will compare action-expert attention for two failed stove paraphrases, a successful stove paraphrase, and the literal stove command. This tests whether the expert ignores the instruction entirely or attends to language tokens whose features encode the wrong task.
```forward
{"type":"probe","model":"original","inputs":["WORK/roll_1_frames/libero_goal_v1385_s7_original/t0000_inputs.pt","WORK/roll_1_frames/libero_goal_v1420_s7_original/t0000_inputs.pt","WORK/roll_1_frames/libero_goal_v1400_s7_original/t0000_inputs.pt","WORK/roll_2_frames/libero_goal_v1785_s7_original/t0000_inputs.pt"],"return":{"layers":[0,6,12,17],"vlm_layers":[0,6,12,17],"what":"attn_maps"}}
```
