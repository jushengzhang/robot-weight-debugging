#!/bin/bash
# stop all pi0.5 policy servers (matches only this script path)
pkill -f "[w]sp_pi05/policy_server.py" && echo stopped
rm -f /mnt/SSD_16TB/jusheng/wsp_pi05/run/sock/*.sock /mnt/SSD_16TB/jusheng/wsp_pi05/run/sock/*.ready
