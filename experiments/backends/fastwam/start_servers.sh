#!/bin/bash
# usage: [START=k] start_servers.sh "2:5 3:5"   (gpu:num_processes ...) -- launches detached servers named g<gpu>_<k..>
F=/mnt/SSD2_16TB/jusheng/wsp/fastwam
W=/mnt/SSD2_16TB/jusheng/Documents/Codex/2026-10-02/vla-wam-cvpr-gemini-api-key/work
PY=/mnt/SSD2_16TB/jusheng/wsp/envs/fastwam_plus/bin/python
mkdir -p $F/run/sock $F/logs
for spec in $1; do
  g=${spec%%:*}; n=${spec##*:}
  for ((k=${START:-0};k<${START:-0}+n;k++)); do
    name=g${g}_${k}
    rm -f $F/run/sock/$name.ready
    CUDA_VISIBLE_DEVICES=$g MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$g PYOPENGL_PLATFORM=egl \
    LIBERO_CONFIG_PATH=$F/libero_cfg MAGICK_HOME=$W/new_benchmark_runtime/magick \
    LD_LIBRARY_PATH=$W/new_benchmark_runtime/magick/lib:${LD_LIBRARY_PATH:-} \
    PYTHONUNBUFFERED=1 setsid nohup $PY $F/policy_server.py --sock $F/run/sock/$name.sock \
      > $F/logs/server_$name.log 2>&1 < /dev/null &
    echo "started $name pid $!"
    sleep 2
  done
done
