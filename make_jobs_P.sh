#!/bin/bash
DD="method.alpha_first=0.7 method.alpha_last=0.3"
for s in 0 1 2; do
  echo "vitb16_TA8_dac_dd_sc1.2-2.4|clip-vit-base-patch16|TA8|dacmerge|$s|$DD method.scale_first=1.2 method.scale_last=2.4"
done
