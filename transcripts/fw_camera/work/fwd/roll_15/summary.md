## arm=edit  episodes=128  errors=0

| category | n | success | published |
|---|---|---|---|
| Light Conditions | 128 | 93.0% | 96.6 |

| suite | n | success | published |
|---|---|---|---|
| libero_10 | 32 | 75.0% |  |
| libero_goal | 33 | 97.0% |  |
| libero_object | 30 | 100.0% |  |
| libero_spatial | 33 | 100.0% |  |
| TOTAL | 128 | 93.0% | 70.8 |

wall/episode: mean 17.2s median 12.4s max 93.5s; mean steps 188
## arm=orig  episodes=128  errors=1

| category | n | success | published |
|---|---|---|---|
| Light Conditions | 127 | 92.9% | 96.6 |

| suite | n | success | published |
|---|---|---|---|
| libero_10 | 32 | 71.9% |  |
| libero_goal | 33 | 100.0% |  |
| libero_object | 30 | 100.0% |  |
| libero_spatial | 32 | 100.0% |  |
| TOTAL | 127 | 92.9% | 70.8 |

wall/episode: mean 16.9s median 12.8s max 81.6s; mean steps 185
ERR libero_spatial v2137: error: FatalError('Offscreen framebuffer is not complete, error 0x8cdd')
