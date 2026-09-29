#!/bin/bash
M=clip-vit-base-patch16
for s in 0 1 2 3 4; do
  echo "vitb16_TA8_opcm|$M|TA8|capmerge|$s|"
  echo "vitb16_TA8_cap_depthdown|$M|TA8|capmerge|$s|method.alpha_schedule=linear method.alpha_first=0.7 method.alpha_last=0.3"
done
