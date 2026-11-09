## arm=orig  episodes=1700  errors=10

| ptype | libero_spatial | libero_object | libero_goal | libero_10 | all | published pi0.5 / pi0 (mean) |
|---|---|---|---|---|---|---|
| original | 96% (50) | 100% (50) | 98% (50) | 90% (50) | 96% (200) |  |
| object | 97% (75) [97/95] | 96% (75) [98/94] | 84% (75) [97/94] | 64% (74) [92/79] | 85% (299) | 96.0 / 90.5 |
| position | 47% (75) [20/0] | 19% (75) [17/0] | 40% (75) [38/0] | 7% (75) [8/0] | 28% (300) | 20.8 / 0.0 |
| semantic | 96% (75) [97/97] | 100% (75) [96/90] | 95% (75) [97/93] | 93% (75) [93/82] | 96% (300) | 95.8 / 90.5 |
| task | 57% (75) [1/0] | 11% (75) [1/0] | 24% (75) [0/0] | 13% (75) [1/0] | 26% (300) | 0.8 / 0.0 |
| environment | 37% (70) [46/60] | 80% (71) [73/29] | 37% (75) [46/39] | 75% (75) [46/27] | 57% (291) | 52.8 / 38.7 |
| ALL | 71% (420) | 66% (421) | 61% (425) | 55% (424) | 63% (1690) |  |

TOTAL 1065/1690 = 63.0%
wall/episode: mean 21.1s median 19.7s max 50.5s
ERR libero_object/environment/pick_up_the_milk_and_place_it_in_the_basket/35: error: FatalError('Offscreen framebuffer is not complete, error 0x8cdd')
ERR libero_object/environment/pick_up_the_butter_and_place_it_in_the_basket/28: error: FatalError('Offscreen framebuffer is not complete, error 0x8cdd')
ERR libero_object/environment/pick_up_the_orange_juice_and_place_it_in_the_basket/21: error: FatalError('Offscreen framebuffer is not complete, error 0x8cdd')
ERR libero_object/environment/pick_up_the_cream_cheese_and_place_it_in_the_basket/05: error: FatalError('Offscreen framebuffer is not complete, error 0x8cdd')
ERR libero_spatial/environment/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate/13: error: FatalError('Offscreen framebuffer is not complete, error 0x8cdd')
