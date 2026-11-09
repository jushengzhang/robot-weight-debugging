#!/bin/bash
# usage: [START=k] start_servers.sh "0:4 1:4 6:4 7:4"   (gpu:num_processes ...) -> detached servers g<gpu>_<k..>
# ONLY GPUs 0,1,6,7 may be used on js186.
# ORIG=1 -> servers on the ORIGINAL LIBERO (unperturbed sanity check), named o<gpu>_<k>
F=/mnt/SSD_16TB/jusheng/wsp_pi05
PY=$F/venv/bin/python
MAGICK=/home/jusheng/anaconda3/envs/erase   # ImageMagick 7.1.2 (read-only use) for LIBERO-Plus Sensor-Noise (Wand)
mkdir -p $F/run/sock $F/logs
for spec in $1; do
  g=${spec%%:*}; n=${spec##*:}
  case $g in 0|1|6|7) ;; *) echo "GPU $g not allowed"; exit 1;; esac
  for ((k=${START:-0};k<${START:-0}+n;k++)); do
    name=g${g}_${k}; CFG=$F/libero_cfg; PP=
    if [ -n "$ORIG" ]; then name=o${g}_${k}; CFG=$F/libero_cfg_orig; PP=$F/third_party/LIBERO-orig; export WSP_ORIG=1; fi
    rm -f $F/run/sock/$name.ready
    CUDA_VISIBLE_DEVICES=$g MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$g PYOPENGL_PLATFORM=egl \
    LIBERO_CONFIG_PATH=$CFG PYTHONPATH=$PP MAGICK_HOME=$MAGICK WAND_MAGICK_LIBRARY_SUFFIX="-7.Q16HDRI" \
    OMP_NUM_THREADS=4 \
    HF_HOME=$F/cache/hf OPENPI_DATA_HOME=$F/cache/openpi \
    PYTHONUNBUFFERED=1 setsid nohup $PY $F/policy_server.py --sock $F/run/sock/$name.sock ${EXTRA:-} \
      > $F/logs/server_$name.log 2>&1 < /dev/null &
    echo "started $name pid $!"
    sleep 2
  done
done
