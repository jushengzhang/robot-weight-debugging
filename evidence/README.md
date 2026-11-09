# Evidence bundle — revised November 5, 2026

Run `python audit_new_results.py` (standard library only) to recompute primary language results, exact paired tests, token-hit partitions, controls with both error conventions, metadata sensitivity and archived PRO outcomes. Raw records are in `raw/`; their original absolute-path provenance is retained in the source manifests. `new-results-verified.json` is the resulting audited table. The original suffix study evidence remains under its original filenames.

All 16 editor transcripts are a separate download. The sparse 65-row edit and application script are also separate. `heldout-tokenization.json` contains the actual SentencePiece token IDs and character offsets used by the web archive viewer. `pi05-video-provenance.jsonl` records the selected paired video episodes. Original source logs are unchanged.
