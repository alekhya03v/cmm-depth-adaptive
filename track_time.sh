#!/bin/bash
# Usage: ./track_time.sh logs/queue_H.log
LOG=${1:-logs/queue_H.log}
echo "== Download: $(grep -c 'MODEL OK' logs/prefetch_b32.log 2>/dev/null)/21 models, $(grep -c 'DATA OK' logs/prefetch_b32.log 2>/dev/null)/20 datasets $(grep -q 'ALL DOWNLOADS DONE' logs/prefetch_b32.log 2>/dev/null && echo '(DONE)')"
[ -f "$LOG" ] || { echo "Runs not started yet"; exit; }
now=$(date +%s); today=$(date +%F)
declare -A START
printf "%-34s %-4s %-8s %s\n" "RUN" "SEED" "STATUS" "DURATION"
while read -r t g what name seed rest; do
  ts=$(date -d "$today $t" +%s); [ $ts -gt $now ] && ts=$((ts-86400))
  key="$name $seed"
  if [ "$what" = "start" ]; then START[$key]=$ts
  elif [ "$what" = "DONE" ] || [ "$what" = "FAILED" ]; then
    d=$(( (ts-${START[$key]:-$ts})/60 )); printf "%-34s %-4s %-8s %dh%02dm\n" "$name" "$seed" "$what" $((d/60)) $((d%60)); unset START[$key]
  fi
done < <(grep -E " (start|DONE|FAILED) " "$LOG")
for key in "${!START[@]}"; do
  d=$(( (now-${START[$key]})/60 )); set -- $key
  printf "%-34s %-4s %-8s %dh%02dm so far\n" "$1" "$2" "RUNNING" $((d/60)) $((d%60))
done
total=$(grep -vc '^\s*$' jobs_H_long.txt); done_=$(grep -c ' DONE ' "$LOG")
echo "== Finished $done_ / $total runs"
