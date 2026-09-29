#!/bin/bash
for M in opcm weight_average task_arithmetic ties_merging symerge_keep-false symerge_keep-true; do
  line=""
  for S in 0 1 2 3 4; do
    n=$(ls outputs/cmm/vitb32_TA8_${M}/version_$S/report_*.json 2>/dev/null | wc -l)
    line="$line s$S:$n/8"
  done
  printf "%-20s %s\n" "$M" "$line"
done
echo "running jobs: $(pgrep -f 'fusion_bench fabric' | wc -l)"
echo "baselines: $(cat logs/baselines_master.log)"
echo "flaw runs: $(cat logs/flaws_master.log)"
