#!/bin/bash
run() { CUDA_VISIBLE_DEVICES=$1 fusion_bench fabric.loggers.root_dir=outputs/cmm \
  fabric.loggers.name=vitb32_TA8_symerge_keep-$2 fabric.loggers.version=$3 \
  method=opcm/symerge method.keep_synergy=$2 method.seed=$3 \
  modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8_model_only \
  taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 > logs/symerge_keep-$2_seed$3.log 2>&1; }
(run 0 false 3; run 0 true 4) & (run 1 true 0) & (run 2 true 2) & (run 3 true 3) &
wait; echo "ALL FLAW RUNS DONE"
