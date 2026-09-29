#!/bin/bash
printf "%-42s %s\n" "EXPERIMENT" "FINISHED ORDERS"
for d in outputs/cmm/*/; do
  n=$(basename "$d")
  case $n in *TALL20*) T=20;; *TALL14*) T=14;; *) T=8;; esac
  done_=$(ls "$d"/version_*/report_$((T-1)).json 2>/dev/null | wc -l)
  total=$(ls -d "$d"/version_* 2>/dev/null | wc -l)
  printf "%-42s %s/%s\n" "$n" "$done_" "$total"
done
echo "---"
echo "GPUs busy: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l) / 4"
tail -n 4 logs/queue_*.log 2>/dev/null
