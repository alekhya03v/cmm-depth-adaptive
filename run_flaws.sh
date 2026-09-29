#!/bin/bash
until grep -q "ALL BASELINES DONE" logs/baselines_master.log 2>/dev/null; do sleep 60; done
JOBS=()
for K in false true; do for S in 0 1 2 3 4; do JOBS+=("$K $S"); done; done
for G in 0 1 2 3; do
  (
    for ((i=G; i<${#JOBS[@]}; i+=4)); do
      set -- ${JOBS[$i]}; K=$1; S=$2
      CUDA_VISIBLE_DEVICES=$G fusion_bench \
        fabric.loggers.root_dir=outputs/cmm \
        fabric.loggers.name=vitb32_TA8_symerge_keep-${K} \
        fabric.loggers.version=$S \
        method=opcm/symerge method.keep_synergy=${K} method.seed=$S \
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
        > logs/symerge_keep-${K}_seed${S}.log 2>&1
    done
  ) &
done
wait
echo "ALL FLAW RUNS DONE"
