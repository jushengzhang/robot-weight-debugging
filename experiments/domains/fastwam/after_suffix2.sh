#!/bin/bash
# waits for the round-2 suffix runs (args: run names), then evaluates each on held-out variants, tagged and clean
W=/mnt/SSD2_16TB/jusheng/wsp; P=$W/envs/kernel/bin/python
for r in "$@"; do
  until python3 -c "import json,sys;sys.exit(0 if json.load(open('$W/runs/$r/state.json')).get('done') else 1)" 2>/dev/null; do sleep 120; done
  rm -rf $W/final/$r; mkdir -p $W/final/$r; cp -r $W/runs/$r/work/edited/. $W/final/$r/
  if [ -z "$(ls $W/final/$r)" ]; then echo "$r: no edit submitted"; continue; fi
  $P $W/harness/eval_heldout.py --per-cat 100 --targets none --language raw --out $W/heldout/${r}_raw --orig-from $W/heldout/${1}_raw --edit $r=$W/final/$r
  $P $W/harness/eval_heldout.py --per-cat 100 --targets none --language strip --out $W/heldout/${r}_strip --orig-from $W/heldout/${1}_strip --edit $r=$W/final/$r
done
echo ALLDONE
