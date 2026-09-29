#!/bin/bash
for B in J K L; do
  L=logs/queue_$B.log
  d=$(grep -c ' DONE ' $L 2>/dev/null); f=$(grep -c ' FAILED ' $L 2>/dev/null)
  r=$(( $(grep -c ' start ' $L 2>/dev/null) - d - f ))
  st=$(grep -q "QUEUE FINISHED" $L 2>/dev/null && echo "FINISHED" || ([ -s $L ] && echo "running" || echo "waiting"))
  echo "Batch $B: $d/18 done, $r running, $f failed  ($st)"
done
echo "--- latest:"; tail -2 logs/queue_J.log; tail -2 logs/queue_K.log 2>/dev/null
