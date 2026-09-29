#!/bin/bash
M=clip-vit-base-patch32
for s in 3 4; do
  echo "vitb32_TA8_cap_depthdown|$M|TA8|capmerge|$s|method.alpha_schedule=linear method.alpha_first=0.7 method.alpha_last=0.3"
done
for pair in "0.6 0.4" "0.8 0.2" "0.9 0.1"; do
  set -- $pair
  for s in 0 1 2; do
    echo "vitb32_TA8_cap_dd$1-$2|$M|TA8|capmerge|$s|method.alpha_schedule=linear method.alpha_first=$1 method.alpha_last=$2"
  done
done
