#!/bin/bash
# usage: (cpu:N = N CPU-inference servers, EGL on GPU ${CPU_EGL_GPU:-6}, ${CPU_THREADS:-16} threads each)
# usage: [START=k] bash start_servers.sh "6:1 7:1"  -> detached recover servers r<gpu>_<k> on ORIGINAL LIBERO
# ONLY GPUs 6,7 are allowed for wsp_recover (0,1 are fully used by wsp_pi05 experiments; 2-5 forbidden).
# Reuses (read-only) the wsp_pi05 venv, checkpoint, caches and LIBERO-orig copy.
R=/mnt/SSD_16TB/jusheng/wsp_recover
F=/mnt/SSD_16TB/jusheng/wsp_pi05
PY=$F/venv/bin/python
mkdir -p $R/run/sock $R/logs
for spec in $1; do
  g=${spec%%:*}; n=${spec##*:}
  DEV=cuda; THR=4
  if [ "$g" = cpu ]; then DEV=cpu; THR=${CPU_THREADS:-16}; g=${CPU_EGL_GPU:-6}; fi   # cpu:N -> model on CPU, EGL render on GPU 6
  case $g in 0|1|6|7) ;; *) echo "GPU $g not allowed"; exit 1;; esac
  [ $DEV = cpu ] || free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $g)
  if [ $DEV = cuda ] && [ "$free" -lt $((9000*n)) ]; then echo "GPU $g has only ${free} MiB free"; exit 1; fi
  for ((k=${START:-0};k<${START:-0}+n;k++)); do
    name=r${g}_${k}; [ $DEV = cpu ] && name=c${g}_${k}
    rm -f $R/run/sock/$name.ready
    CUDA_VISIBLE_DEVICES=$g MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$g PYOPENGL_PLATFORM=egl \
    WSP_PI05=$F WSP_ORIG=1 LIBERO_CONFIG_PATH=$R/libero_cfg_orig PYTHONPATH=$F/third_party/LIBERO-orig \
    WSP_DEVICE=$DEV OMP_NUM_THREADS=$THR MKL_NUM_THREADS=$THR HF_HOME=$F/cache/hf OPENPI_DATA_HOME=$F/cache/openpi \
    TORCHINDUCTOR_CACHE_DIR=$R/cache/inductor XDG_CACHE_HOME=$R/cache TMPDIR=$R/cache/tmp \
    PYTHONUNBUFFERED=1 setsid nohup $PY $R/recover_server.py --sock $R/run/sock/$name.sock ${EXTRA:-} \
      > $R/logs/server_$name.log 2>&1 < /dev/null &
    echo "started $name pid $!"; echo $! >> $R/run/pids
    sleep 2
  done
done
