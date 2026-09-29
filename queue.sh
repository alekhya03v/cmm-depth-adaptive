#!/bin/bash
# Resumable GPU job queue. Usage: ./queue.sh jobs_xxx.txt
# Each job line: NAME|MODEL|SUITE|METHOD|SEED|EXTRA_OVERRIDES
JOBS=$1
[ -f "$JOBS" ] || { echo "no job file $JOBS"; exit 1; }
mapfile -t L < <(grep -v '^#' "$JOBS" | grep -v '^\s*$')
N=${#L[@]}
CNT=logs/.counter_$(basename "$JOBS"); echo 0 > "$CNT"
next_idx() { ( flock -x 9; i=$(cat "$CNT"); echo $((i+1)) > "$CNT"; echo "$i" ) 9>"$CNT.lock"; }
for G in ${GPUS:-0 1 2 3}; do
  (
    while true; do
      i=$(next_idx); [ "$i" -ge "$N" ] && break
      IFS='|' read -r NAME MODEL SUITE METHOD SEED EXTRA <<< "${L[$i]}"
      case $SUITE in TA8) T=8;; TALL14) T=14;; TALL20) T=20;; esac
      OUT=outputs/cmm/$NAME/version_$SEED
      if [ -f "$OUT/report_$((T-1)).json" ]; then echo "skip $NAME s$SEED"; continue; fi
      rm -rf "$OUT"
      echo "$(date +%H:%M) GPU$G start $NAME s$SEED"
      CUDA_VISIBLE_DEVICES=$G ${FB:-fusion_bench} \
        fabric.loggers.root_dir=outputs/cmm fabric.loggers.name=$NAME fabric.loggers.version=$SEED \
        method=opcm/$METHOD method.seed=$SEED method.save_on_every_step=false $EXTRA \
        modelpool=CLIPVisionModelPool/${MODEL}_${SUITE}_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_$SUITE \
        taskpool.base_model=openai/$MODEL \
        > logs/${NAME}_s${SEED}.log 2>&1
      if [ -f "$OUT/report_$((T-1)).json" ]; then echo "$(date +%H:%M) GPU$G DONE $NAME s$SEED"
      else echo "$(date +%H:%M) GPU$G FAILED $NAME s$SEED -> see logs/${NAME}_s${SEED}.log"; fi
    done
  ) &
done
wait
echo "QUEUE FINISHED: $JOBS"
