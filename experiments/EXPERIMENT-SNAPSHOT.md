# Weight-space debugging of robot policies: code and paper materials

Latest code for all experiments, collected on 2026-11-05.
The live copies the experiments ran from are:
- host 120: /mnt/SSD2_16TB/jusheng/wsp/harness (FastWAM) and wsp/fastwam (backend)
- host 186: /mnt/SSD_16TB/jusheng/wsp_pi05/harness (pi0.5 domains), wsp_pi05 / wsp_recover / wsp_pro (backends)

| dir | contents |
|---|---|
| harness/ | driver.py (LLM loop, budget, handoff, final-turn cleanup), kernel.py (sandboxed CPU Python kernel) |
| domains/fastwam/ | task prompts + forward tool + held-out eval for FastWAM on LIBERO-Plus (camera/background/language, instruction-suffix) |
| domains/pi05/ | task prompts + forward tools + held-out evals for pi0.5: LIBERO-Plus Language (domain_pi05), LIBERO-Recovery (domain_recover), LIBERO-PRO (domain_pro) |
| backends/ | policy servers/clients (same socket protocol: orig/edit arms, weight override with verified restore, activation record, frames, probe) |
| paper/ | additions.tex (LaTeX blocks to merge into the Nov 4 draft), make_figs.py, figs/ |

Before release: remove host-specific absolute paths, add a config file, add LICENSE, and strip nothing secret (no keys are stored in these files; run_astra*.sh read keys from ~/.config/llmkeys at run time).
