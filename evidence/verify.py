"""Recompute the aggregate paired statistics from the included reduced records.

Run with Python 3; uses only the standard library and writes no files.
"""
import json
import math
from pathlib import Path

root = Path(__file__).resolve().parent
records = [json.loads(line) for line in (root / 'episode-pairs.jsonl').open()]
reference = json.loads((root / 'verified-results.json').read_text())
for condition, field in [('raw', 'raw_heldout'), ('clean', 'clean_heldout')]:
    rows = [r for r in records if r['condition'] == condition]
    keys = [(r['suite'], r['variant_id'], r['seed'], r['init_idx']) for r in rows]
    assert len(set(keys)) == len(rows), 'duplicate pair'
    n = len(rows)
    original = sum(r['original_success'] for r in rows)
    candidate = sum(r['candidate_success'] for r in rows)
    gain = sum(not r['original_success'] and r['candidate_success'] for r in rows)
    loss = sum(r['original_success'] and not r['candidate_success'] for r in rows)
    p = min(1.0, 2 * sum(math.comb(gain + loss, i) for i in range(min(gain, loss) + 1)) / 2 ** (gain + loss))
    expected = reference[field]['ALL']
    assert (n, original, candidate, gain, loss) == tuple(expected[k] for k in ['n', 'original_successes', 'candidate_successes', 'gain_pairs', 'loss_pairs'])
    assert abs(p - expected['mcnemar_exact_p']) < 1e-12
    print(f'{condition}: {original}/{n} -> {candidate}/{n}; gains {gain}, losses {loss}; delta {(candidate-original)/n*100:+.4f} pp; exact p={p:.10g}')
print('All aggregate statistics match the accompanying verified results.')
