## arm=edit  episodes=1700  errors=7

| ptype | libero_spatial | libero_object | libero_goal | libero_10 | all | published pi0.5 / pi0 (mean) |
|---|---|---|---|---|---|---|
| original | 98% (50) | 100% (50) | 94% (50) | 94% (50) | 96% (200) |  |
| object | 97% (75) [97/95] | 97% (75) [98/94] | 89% (75) [97/94] | 74% (73) [92/79] | 90% (298) | 96.0 / 90.5 |
| position | 44% (75) [20/0] | 24% (75) [17/0] | 36% (75) [38/0] | 5% (75) [8/0] | 27% (300) | 20.8 / 0.0 |
| semantic | 97% (75) [97/97] | 100% (75) [96/90] | 97% (75) [97/93] | 95% (75) [93/82] | 97% (300) | 95.8 / 90.5 |
| task | 61% (75) [1/0] | 11% (75) [1/0] | 27% (75) [0/0] | 15% (75) [1/0] | 28% (300) | 0.8 / 0.0 |
| environment | 31% (72) [46/60] | 79% (73) [73/29] | 43% (75) [46/39] | 76% (75) [46/27] | 57% (295) | 52.8 / 38.7 |
| ALL | 70% (422) | 67% (423) | 63% (425) | 58% (423) | 64% (1693) |  |

TOTAL 1088/1693 = 64.3%
wall/episode: mean 20.8s median 19.3s max 77.1s
ERR libero_object/environment/pick_up_the_ketchup_and_place_it_in_the_basket/22: error: FatalError('Offscreen framebuffer is not complete, error 0x8cdd')
ERR libero_object/environment/pick_up_the_ketchup_and_place_it_in_the_basket/23: error: FatalError('Offscreen framebuffer is not complete, error 0x8cdd')
ERR libero_spatial/environment/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate/36: error: FatalError('Offscreen framebuffer is not complete, error 0x8cdd')
ERR libero_spatial/environment/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate/01: error: FatalError('Offscreen framebuffer is not complete, error 0x8cdd')
ERR libero_spatial/environment/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate/12: error: FatalError('Offscreen framebuffer is not complete, error 0x8cdd')
