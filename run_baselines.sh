#!/bin/bash
# 15 jobs = 3 methods x 5 seeds, split across 4 GPUs (each GPU runs its share one after another)
JOBS=()
for M in weight_average task_arithmetic ties_merging; do
  for S in 0 1 2 3 4; do JOBS+=("$M $S"); done
done
for G in 0 1 2 3; do
  (
    for ((i=G; i<${#JOBS[@]}; i+=4)); do
      set -- ${JOBS[$i]}; M=$1; S=$2
      EXTRA=""
      [ "$M" = "ties_merging" ] && EXTRA="method.scaling_factor=0.3"
      CUDA_VISIBLE_DEVICES=$G fusion_bench \
        fabric.loggers.root_dir=outputs/cmm \
        fabric.loggers.name=vitb32_TA8_${M} \
        fabric.loggers.version=$S \
        method=opcm/${M} method.seed=$S method.save_on_every_step=false $EXTRA \
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
        > logs/${M}_seed${S}.log 2>&1
    done
  ) &
done
wait
echo "ALL BASELINES DONE"
