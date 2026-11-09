## arm=edit  episodes=765  errors=3

| category | n | success | published pi0 |
|---|---|---|---|
| Language Instructions | 762 | 85.6% | 58.8 |

| suite | n | success | published pi0 |
|---|---|---|---|
| libero_10 | 178 | 91.0% |  |
| libero_goal | 206 | 70.9% |  |
| libero_object | 187 | 94.7% |  |
| libero_spatial | 191 | 87.4% |  |
| TOTAL | 762 | 85.6% | 53.6 |

wall/episode: mean 15.1s median 13.1s max 40.5s; mean steps 180
ERR libero_goal v1103: error: FatalError('Offscreen framebuffer is not complete, error 0x8cdd')
ERR libero_object v1141: error: FatalError('Offscreen framebuffer is not complete, error 0x8cdd')
ERR libero_10 v1381: error: FatalError('Offscreen framebuffer is not complete, error 0x8cdd')
