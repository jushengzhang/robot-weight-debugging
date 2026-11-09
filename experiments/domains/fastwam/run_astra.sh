#!/bin/bash
# usage: run_astra.sh <run_name> "<target category>" [passes]
# env DOMAIN=<module.py> selects another domain (default domain_fastwam.py)
# waits for the converted weights, then runs the driver; restarts it (resume) if it exits without finishing
W=/mnt/SSD2_16TB/jusheng/wsp; R=$W/runs/$1; export WSP_TARGET="$2"; P=${3:-40}
mkdir -p $R
until [ -s $W/fastwam/weights/fastwam_libero_uncond.safetensors ] && grep -q "^saved" $W/fastwam/weights/convert.log 2>/dev/null; do sleep 30; done
set -a; . ~/.config/llmkeys/openai_astra.env; set +a
for i in $(seq 1 50); do
  $W/envs/kernel/bin/python $W/harness/driver.py --run $R --domain $W/harness/${DOMAIN:-domain_fastwam.py} \
    --model gpt-6-astra --effort max --passes $P >> $R/stdout.log 2>&1
  python3 -c "import json,sys;sys.exit(0 if json.load(open('$R/state.json')).get('done') else 1)" 2>/dev/null && break
  echo "$(date +%F_%T) driver exited, restarting" >> $R/stdout.log; sleep 60
done
