#!/bin/bash
cd ~/alekh_opcm/opcm
git add -A fusion_bench fusion_bench_config *.py *.sh results/paper 2>/dev/null
git commit -qm "autosave $(date +%F_%H:%M)" && git push -q
