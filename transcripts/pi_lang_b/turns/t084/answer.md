I’ll replay the same 16 observations with the direction-only control, recording VLM and action-expert residuals as well as actions. Holding images and flow noise fixed isolates the effect of embedding direction.
```forward
{
  "type": "probe",
  "model": "both",
  "inputs": [
    "/mnt/SSD_16TB/jusheng/wsp_pi05/runs_wsp/pi_lang_b/work/fwd/roll_1/artifacts/libero_goal_v1415_s7_i0_orig_frames/t0000_inputs.pt",
    "/mnt/SSD_16TB/jusheng/wsp_pi05/runs_wsp/pi_lang_b/work/fwd/roll_1/artifacts/libero_goal_v1416_s7_i0_orig_frames/t0000_inputs.pt",
    "/mnt/SSD_16TB/jusheng/wsp_pi05/runs_wsp/pi_lang_b/work/fwd/roll_1/artifacts/libero_goal_v1400_s7_i0_orig_frames/t0000_inputs.pt",
    "/mnt/SSD_16TB/jusheng/wsp_pi05/runs_wsp/pi_lang_b/work/fwd/roll_2/artifacts/libero_goal_v1390_s7_i0_orig_frames/t0000_inputs.pt",
    "/mnt/SSD_16TB/jusheng/wsp_pi05/runs_wsp/pi_lang_b/work/fwd/roll_1/artifacts/libero_goal_v1179_s7_i0_orig_frames/t0000_inputs.pt",
    "/mnt/SSD_16TB/jusheng/wsp_pi05/runs_wsp/pi_lang_b/work/fwd/roll_2/artifacts/libero_goal_v1169_s7_i0_orig_frames/t0000_inputs.pt",
    "/mnt/SSD_16TB/jusheng/wsp_pi05/runs_wsp/pi_lang_b/work/fwd/roll_1/artifacts/libero_goal_v1339_s7_i0_orig_frames/t0000_inputs.pt",
    "/mnt/SSD_16TB/jusheng/wsp_pi05/runs_wsp/pi_lang_b/work/fwd/roll_1/artifacts/libero_goal_v1341_s7_i0_orig_frames/t0000_inputs.pt",
    "/mnt/SSD_16TB/jusheng/wsp_pi05/runs_wsp/pi_lang_b/work/fwd/roll_2/artifacts/libero_goal_v1378_s7_i0_orig_frames/t0000_inputs.pt",
    "/mnt/SSD_16TB/jusheng/wsp_pi05/runs_wsp/pi_lang_b/work/fwd/roll_1/artifacts/libero_goal_v1263_s7_i0_orig_frames/t0000_inputs.pt",
    "/mnt/SSD_16TB/jusheng/wsp_pi05/runs_wsp/pi_lang_b/work/fwd/roll_2/artifacts/libero_goal_v1278_s7_i0_orig_frames/t0000_inputs.pt",
    "/mnt/SSD_16TB/jusheng/wsp_pi05/runs_wsp/pi_lang_b/work/fwd/roll_2/artifacts/libero_goal_v1431_s7_i0_orig_frames/t0000_inputs.pt",
    "/mnt/SSD_16TB/jusheng/wsp_pi05/runs_wsp/pi_lang_b/work/fwd/roll_1/artifacts/libero_goal_v1331_s7_i0_orig_frames/t0000_inputs.pt",
    "/mnt/SSD_16TB/jusheng/wsp_pi05/runs_wsp/pi_lang_b/work/fwd/roll_1/artifacts/libero_goal_v1462_s7_i0_orig_frames/t0000_inputs.pt",
    "/mnt/SSD_16TB/jusheng/wsp_pi05/runs_wsp/pi_lang_b/work/fwd/roll_4/artifacts/libero_object_v1376_s7_i0_orig_frames/t0000_inputs.pt",
    "/mnt/SSD_16TB/jusheng/wsp_pi05/runs_wsp/pi_lang_b/work/fwd/roll_4/artifacts/libero_object_v1377_s7_i0_orig_frames/t0000_inputs.pt"
  ],
  "return": {
    "layers": [0, 3, 6, 9, 12, 15, 17],
    "vlm_layers": [0, 3, 6, 9, 12, 15, 17],
    "what": "resid"
  }
}
```
