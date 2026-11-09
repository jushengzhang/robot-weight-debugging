# Exp 8 ("Sparks of Weight-Space Programming") — independent check of the p50c MATH-500 claim

Host 120, 2026-11-03 (CST). By Claude, an agent working for jusheng.

## Verdict
The headline result **reproduces**. The p50c weight edit (2 MLP down_proj tensors, 130,188 bf16 entries) takes
Qwen3.5-0.8B (thinking on) from about 9% to about 49% on MATH-500. It also cuts the share of answers that hit the
32k-token limit from about 92% to 30%.

| MATH-500, think35 protocol         | ours acc | ours cut@32k | paper acc | paper cut@32k |
|---|---|---|---|---|
| unedited, all 500 (paper: 4 samples) | n/a (stopped, see below) | n/a | 7.9 | 93.0 |
| unedited, fixed random 100 (seed 0), 1 sample | **9.0** | **92.0** | 8.8 (same 100, 4 samples) | n/a |
| p50c, all 500, 1 sample            | **49.4** | **30.0** | 48.1 | 33.1 |
| p50c, same random 100, 1 sample    | 57.0 | n/a | 50.2 (same 100, 4 samples) | n/a |

- p50c mean length: 21.8k tokens (paper: 22.0k). Unedited subset: 30.7k (paper, full set: 30.9k).
- p50c per-problem agreement with the paper's released per-problem file: correlation 0.72 between our single sample
  and the paper's 4-sample mean. We solved 136 of the 153 problems the paper solved 4/4, and 11 of the 180 problems it
  solved 0/4.
- With n=500 and 1 sample, the standard error is about 2.2pp, so the 49.4 vs 48.1 and 30.0 vs 33.1 gaps are within noise.
  The sub-100 gap (57 vs 50) is about 1.4 SE.
- GSM8K was **not run**: GPU time was reclaimed for FastWAM.

## What was done
1. Downloaded `Qwen/Qwen3.5-0.8B` @ `2fc06364715b967f1860aea9cf38778875588b17`. The single shard's sha256
   `04b1c301…e4696` matches the HF LFS hash; most of it came through hf-mirror.com.
2. Built p50c with the release's `code/apply_edit.py`, after reading it; it uses numpy only. Both base-tensor and
   edited-tensor sha256 checks passed: `layers.15.mlp.down_proj` (65,260 entries) and `layers.19.mlp.down_proj`
   (64,928 entries). Output: the full multimodal release directory with the shard replaced.
3. Evaluation: a re-implementation of `code/evaluate.py run --protocol think35` (`wsp/scripts/ev.py`). It uses the
   same prompt (problem + "\nPlease reason step by step, and put your final answer within \boxed{}."), the chat
   template with `enable_thinking=True`, and the same sampling: T 0.6, top_p 0.95, top_k 20, presence 1.5,
   max_tokens 32768, seed 0. Grading imports the release's own `mathqa.extract`/`is_correct` (last `\boxed{}` after
   `</think>`, math-verify 0.9.0). It ran on vLLM 0.25.1, bf16, one engine per GPU at gpu_mem 0.85.

## Differences from the paper's setup
- **Samples:** 1 per problem instead of 4.
- **Model export:** the paper used a text-only export (`Qwen3_5ForCausalLM`). We ran the full
  `Qwen3_5ForConditionalGeneration` release in vLLM text-only mode (`limit_mm_per_prompt=0`). The language-model
  tensors and the chat template are identical; we diffed the template against the release's `base_model/` copy.
- **Unedited baseline:** we did not run all 500 problems. vLLM's default `max_num_seqs=1024` admitted all 500
  sequences, and almost all of them grow to 32k tokens, so they did not fit in the KV cache (about 271 full-length
  sequences). After 1.6 h no capped sequence had finished, the run was stopped to free GPU 0 for FastWAM, and the
  partial outputs were lost. The fixed random 100-problem subset (`bench/math500sub100_idx.json`) then ran in 41 min.

## Wall time
- Download: about 40 min, mostly because the proxy was slow (about 0.5-0.8 MB/s) until the switch to hf-mirror.
- p50c MATH-500: 2 h 24 min of generation (20:45 to 23:09 CST, GPU 1).
- Unedited 100-problem subset: 41 min (GPU 1).
- Aborted unedited full run: about 2 h on GPU 0.
- Total: about 4 h 15 min (19:50 to 23:52 CST).

## Paths (host 120, under /mnt/SSD2_16TB/jusheng/wsp)
- Models: `models/Qwen3.5-0.8B` (base) and `models/Qwen3.5-0.8B-p50c` (edited, hash-verified).
- Results: `eval/p50c/math500_{scores.json,per_problem.json,samples.jsonl.gz}` and
  `eval/base/math500sub100_{…}` (full texts are included).
- Scripts: `scripts/{ev.py,build.sh,cmp.py,run_eval.sh,after2.sh}`.
- Logs: `eval/logs/`.
- Private packages (no shared env was modified): `pkgs/mathverify`, `pkgs/pyarrow`, `pkgs/pyspy`.
