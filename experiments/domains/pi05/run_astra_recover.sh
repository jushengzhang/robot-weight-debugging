#!/bin/bash
# usage: run_astra.sh <run_name> "<target category>" [passes]   (pi0.5 on js186)
P=/mnt/SSD_16TB/jusheng/wsp_pi05; R=$P/runs_recover/$1; export WSP_TARGET="$2"; N=${3:-50}
mkdir -p $R
set -a; . ~/.config/llmkeys/openai_astra.env; set +a
export http_proxy=http://127.0.0.2:10809 https_proxy=http://127.0.0.2:10809 HTTP_PROXY=http://127.0.0.2:10809 HTTPS_PROXY=http://127.0.0.2:10809 no_proxy=localhost,127.0.0.1
for i in $(seq 1 50); do
  $P/venv/bin/python $P/harness/driver.py --run $R --domain $P/harness/domain_recover.py \
    --model gpt-6-astra --effort max --passes $N >> $R/stdout.log 2>&1
  python3 -c "import json,sys;sys.exit(0 if json.load(open('$R/state.json')).get('done') else 1)" 2>/dev/null && break
  echo "$(date +%F_%T) driver exited, restarting" >> $R/stdout.log; sleep 60
done
