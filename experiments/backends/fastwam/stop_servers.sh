#!/bin/bash
# stop all policy servers started by start_servers.sh (matches only our script path)
pkill -f "[w]sp/fastwam/policy_server.py" && echo stopped
rm -f /mnt/SSD2_16TB/jusheng/wsp/fastwam/run/sock/*.sock /mnt/SSD2_16TB/jusheng/wsp/fastwam/run/sock/*.ready
