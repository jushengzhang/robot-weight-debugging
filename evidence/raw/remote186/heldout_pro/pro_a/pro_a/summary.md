## arm=edit  episodes=700  errors=8

| ptype | libero_spatial | libero_object | libero_goal | libero_10 | all | published pi0.5 / pi0 (mean) |
|---|---|---|---|---|---|---|
| original | 98% (50) | 100% (50) | 94% (50) | 94% (50) | 96% (200) |  |
| object | 92% (25) [97/95] | 96% (25) [98/94] | 88% (25) [97/94] | 80% (25) [92/79] | 89% (100) | 96.0 / 90.5 |
| position | 44% (25) [20/0] | 12% (25) [17/0] | 36% (25) [38/0] | 20% (25) [8/0] | 28% (100) | 20.8 / 0.0 |
| semantic | 100% (25) [97/97] | 100% (25) [96/90] | 96% (25) [97/93] | 88% (25) [93/82] | 96% (100) | 95.8 / 90.5 |
| task | 36% (25) [1/0] | 12% (25) [1/0] | 20% (25) [0/0] | 36% (25) [1/0] | 26% (100) | 0.8 / 0.0 |
| environment | 33% (24) [46/60] | 74% (23) [73/29] | 40% (20) [46/39] | 76% (25) [46/27] | 57% (92) | 52.8 / 38.7 |
| ALL | 72% (174) | 71% (173) | 68% (170) | 70% (175) | 70% (692) |  |

TOTAL 484/692 = 69.9%
wall/episode: mean 19.7s median 17.8s max 58.4s
ERR libero_goal/environment/put_the_bowl_on_top_of_the_cabinet/36: error: FatalError('Offscreen framebuffer is not complete, error 0x8cdd')
ERR libero_goal/environment/put_the_bowl_on_the_stove/26: error: FatalError('Offscreen framebuffer is not complete, error 0x8cdd')
ERR libero_goal/environment/put_the_wine_bottle_on_top_of_the_cabinet/11: error: FatalError('Offscreen framebuffer is not complete, error 0x8cdd')
ERR libero_goal/environment/put_the_wine_bottle_on_the_rack/30: error: FatalError('Offscreen framebuffer is not complete, error 0x8cdd')
ERR libero_goal/environment/put_the_bowl_on_top_of_the_cabinet/26: error: FatalError('Offscreen framebuffer is not complete, error 0x8cdd')
