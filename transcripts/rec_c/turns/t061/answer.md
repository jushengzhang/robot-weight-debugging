I will verify that query softening actually broadens spatial attention and increases access to the base view on the fixed diagnostic observations. If it instead sends attention into language or action-token sinks, it would not test the intended mechanism.
```forward
{"type":"probe","model":"both","inputs":["/mnt/SSD_16TB/jusheng/wsp_pi05/runs_recover/rec_c/work/diag4/0.pt","/mnt/SSD_16TB/jusheng/wsp_pi05/runs_recover/rec_c/work/diag4/1.pt","/mnt/SSD_16TB/jusheng/wsp_pi05/runs_recover/rec_c/work/diag4/2.pt","/mnt/SSD_16TB/jusheng/wsp_pi05/runs_recover/rec_c/work/diag4/3.pt"],"return":{"layers":[2,8,14,15],"what":"attn_maps"}}
```
