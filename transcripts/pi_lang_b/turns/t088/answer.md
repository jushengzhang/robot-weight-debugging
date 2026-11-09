All four successes lost under the destination ablation involve “storage unit” descriptions of the cabinet, which supports the proposed destination-confusion mechanism. The validated edit is restored. I’ll now compare attention maps for the cooktop and wine/cabinet paraphrases and their successful controls, checking whether the failures stem from missing language attention or from the features that attention retrieves.
```forward
{
  "type": "probe",
  "model": "both",
  "inputs": [
    "/mnt/SSD_16TB/jusheng/wsp_pi05/runs_wsp/pi_lang_b/work/fwd/roll_1/artifacts/libero_goal_v1415_s7_i0_orig_frames/t0000_inputs.pt",
    "/mnt/SSD_16TB/jusheng/wsp_pi05/runs_wsp/pi_lang_b/work/fwd/roll_2/artifacts/libero_goal_v1390_s7_i0_orig_frames/t0000_inputs.pt",
    "/mnt/SSD_16TB/jusheng/wsp_pi05/runs_wsp/pi_lang_b/work/fwd/roll_1/artifacts/libero_goal_v1179_s7_i0_orig_frames/t0000_inputs.pt",
    "/mnt/SSD_16TB/jusheng/wsp_pi05/runs_wsp/pi_lang_b/work/fwd/roll_2/artifacts/libero_goal_v1169_s7_i0_orig_frames/t0000_inputs.pt"
  ],
  "return": {
    "layers": [0, 3, 9, 15],
    "what": "attn_maps"
  }
}
```
