#!/bin/bash
M=clip-vit-base-patch32; DD="method.alpha_first=0.7 method.alpha_last=0.3"
for s in 0 1 2; do
  echo "vitb32_TA8_dac_dd_sc1.0-2.6|$M|TA8|dacmerge|$s|$DD method.scale_first=1.0 method.scale_last=2.6"
  echo "vitb32_TA8_dac_dd_sc0.8-2.8|$M|TA8|dacmerge|$s|$DD method.scale_first=0.8 method.scale_last=2.8"
done
