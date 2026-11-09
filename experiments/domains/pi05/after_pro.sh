#!/bin/bash
P=/mnt/SSD_16TB/jusheng/wsp_pi05
for r in "$@"; do
  until python3 -c "import json,sys;sys.exit(0 if json.load(open(\"$P/runs_pro/$r/state.json\")).get(\"done\") else 1)" 2>/dev/null; do sleep 120; done
  rm -rf $P/final/$r; mkdir -p $P/final/$r; cp -r $P/runs_pro/$r/work/edited/. $P/final/$r/ 2>/dev/null
  if [ -z "$(ls $P/final/$r)" ]; then echo "$r: no edit submitted"; continue; fi
  $P/venv/bin/python $P/harness/eval_pro.py --cap 25 --out $P/heldout_pro/$r --orig-from $P/heldout_pro/$1 --edit $r=$P/final/$r
done
echo ALLDONE
