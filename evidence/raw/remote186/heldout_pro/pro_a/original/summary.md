## arm=orig  episodes=700  errors=4

| ptype | libero_spatial | libero_object | libero_goal | libero_10 | all | published pi0.5 / pi0 (mean) |
|---|---|---|---|---|---|---|
| original | 96% (50) | 100% (50) | 98% (50) | 88% (50) | 96% (200) |  |
| object | 100% (25) [97/95] | 88% (25) [98/94] | 80% (25) [97/94] | 68% (25) [92/79] | 84% (100) | 96.0 / 90.5 |
| position | 36% (25) [20/0] | 8% (25) [17/0] | 36% (25) [38/0] | 16% (25) [8/0] | 24% (100) | 20.8 / 0.0 |
| semantic | 100% (25) [97/97] | 100% (25) [96/90] | 88% (25) [97/93] | 88% (25) [93/82] | 94% (100) | 95.8 / 90.5 |
| task | 32% (25) [1/0] | 16% (25) [1/0] | 24% (25) [0/0] | 20% (25) [1/0] | 23% (100) | 0.8 / 0.0 |
| environment | 44% (25) [46/60] | 71% (21) [73/29] | 28% (25) [46/39] | 64% (25) [46/27] | 51% (96) | 52.8 / 38.7 |
| ALL | 72% (175) | 69% (171) | 65% (175) | 62% (175) | 67% (696) |  |

TOTAL 465/696 = 66.8%
wall/episode: mean 20.1s median 17.3s max 56.4s
ERR libero_object/environment/pick_up_the_milk_and_place_it_in_the_basket/17: error: FatalError('Offscreen framebuffer is not complete, error 0x8cdd')
ERR libero_object/environment/pick_up_the_bbq_sauce_and_place_it_in_the_basket/30: error: FatalError('Offscreen framebuffer is not complete, error 0x8cdd')
ERR libero_object/environment/pick_up_the_bbq_sauce_and_place_it_in_the_basket/08: error: FatalError('Offscreen framebuffer is not complete, error 0x8cdd')
ERR libero_object/environment/pick_up_the_alphabet_soup_and_place_it_in_the_basket/10: error: FatalError('Offscreen framebuffer is not complete, error 0x8cdd')
