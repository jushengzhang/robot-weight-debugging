# Evidence behind the reported numbers

This directory contains episode records and statistical audits for the first
research release. Start with the relevant question below, or read the
[results and limitations](../docs/RESULTS.md).

| Claim or question | Records / calculation |
| --- | --- |
| Three language edits on 765 held-out episodes | [Held-out summary](raw/remote186/heldout/lang_all/summary.md) · [Audit output](new-results-verified.json) → `pi05_language` |
| 65 rows, affected tokens, and the 335 / 430 split | [Embedding recipe](pi05-lang-c-embedding-recipe.json) · [Actual tokenization](heldout-tokenization.json) · audit output → `touched` / `untouched` |
| Random directions and full lexical replacement | [Control summary](raw/remote186/heldout/lang_ctrl/summary.md) · audit output → `controls` |
| Additional seeds, locality, and norm/direction probes | [Extended records and audit](extended) |
| FastWAM instruction-metadata sensitivity | Audit output → `suffix` |
| First FastWAM visual/language searches | [Wave-one summary](raw/fastwam/heldout/wave1/summary.md) |
| First suffix candidate and follow-up C | [First candidate](raw/fastwam/heldout/suffix_raw/summary.md) · [Follow-up](raw/fastwam/heldout/fw_suffix2_c_raw/summary.md) |
| Archived LIBERO-PRO candidate A | [Larger evaluation](raw/remote186/heldout_pro/pro_a_big/summary.md) · audit output → `pro_archived` |
| Actual paired videos shown in the README | [Episode provenance](pi05-video-provenance.jsonl) · [Rendering and source hashes](../assets) |

## Recompute from the released records

From the repository root:

```bash
python3 evidence/audit_new_results.py
python3 -m pip install 'numpy>=1.26,<3'
python3 evidence/extended/audit_extended.py
```

The first script uses only Python's standard library. It recomputes language
results, exact paired tests, token-hit partitions, controls under both error
conventions, metadata sensitivity, and archived PRO outcomes. The second script
checks the extended evidence. Neither launches a simulator or consumes GPU
inference. The resulting tables are [new-results-verified.json](new-results-verified.json)
and [extended-evidence-verified.json](extended/extended-evidence-verified.json).

## How to interpret the records

Raw records are in `raw/`; original absolute-path provenance is preserved in
source manifests. The original suffix-study evidence is also retained under
its original filenames. An archived candidate's evaluation is distinct from
the editor's decision to retain it. See [all 16 session guides](../transcripts)
for final answers, notes, and executed tool logs. The application-ready sparse
payload is under [edits/pi05-language-c](../edits/pi05-language-c).
