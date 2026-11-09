#!/bin/bash
P=/mnt/SSD_16TB/jusheng/wsp_pi05
for r in "$@"; do
  until python3 -c "import json,sys;sys.exit(0 if json.load(open(\"$P/runs_recover/$r/state.json\")).get(\"done\") else 1)" 2>/dev/null; do sleep 120; done
  rm -rf $P/final/$r; mkdir -p $P/final/$r; cp -r $P/runs_recover/$r/work/edited/. $P/final/$r/ 2>/dev/null
  if [ -z "$(ls $P/final/$r)" ]; then echo "$r: no edit submitted"; continue; fi
  $P/venv/bin/python $P/harness/eval_recover.py --cap 25 --out $P/heldout_recover/$r --orig-from $P/heldout_recover/$1 --edit $r=$P/final/$r
  $P/venv/bin/python $P/harness/libero_check.py $r $P/final/$r $P/heldout_recover/${r}_libero
done
echo ALLDONE
