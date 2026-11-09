The original value tensor is already exactly bf16-representable, so simple re-quantization is not the explanation. I will replay the identical bf16 overlay on the same observations; exact agreement would rule out a loader-path artifact for this candidate before the dose-response test.
```forward
{
 "type":"probe",
 "model":"both",
 "inputs":[
  "WORK/roll_1_frames/s11_close_seed17_original/t0250_inputs.pt",
  "WORK/roll_1_frames/s11_open_seed17_original/t0050_inputs.pt",
  "WORK/roll_1_frames/s935_close_seed17_original/t0250_inputs.pt",
  "WORK/roll_1_frames/s935_open_seed17_original/t0050_inputs.pt",
  "WORK/fwd/roll_1/artifacts/libero_object__pick_up_the_bbq_sauce_and_place_it_in_the_basket__episode_016__step0160_open_s17_orig_frames/t0250_inputs.pt",
  "WORK/fwd/roll_1/artifacts/libero_10__LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket__episode_001__step0300_open_s17_orig_frames/t0250_inputs.pt",
  "WORK/fwd/roll_1/artifacts/libero_goal__put_the_wine_bottle_on_the_rack__episode_007__step0080_open_s17_orig_frames/t0050_inputs.pt",
  "WORK/fwd/roll_1/artifacts/libero_spatial__pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate__episode_022__step0060_open_s17_orig_frames/t0050_inputs.pt"
 ],
 "return":{"layers":[17],"what":"resid"}
}
```
