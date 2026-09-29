#!/bin/bash
M=clip-vit-base-patch32; DD="method.alpha_first=0.7 method.alpha_last=0.3"
for s in 0 1 2; do
  echo "vitb32_TA8_dac_dd_sc1.6-2.0|$M|TA8|dacmerge|$s|$DD method.scale_first=1.6 method.scale_last=2.0"
  echo "vitb32_TA8_dac_dd_sc1.4-2.2|$M|TA8|dacmerge|$s|$DD method.scale_first=1.4 method.scale_last=2.2"
  echo "vitb32_TA8_dac_dd_sc1.2-2.4|$M|TA8|dacmerge|$s|$DD method.scale_first=1.2 method.scale_last=2.4"
  echo "vitb32_TA8_dac_dd_sc1.6-2.4|$M|TA8|dacmerge|$s|$DD method.scale_first=1.6 method.scale_last=2.4"
  echo "vitb32_TA8_dac_sc1.4-1.8|$M|TA8|dacmerge|$s|method.scale_first=1.4 method.scale_last=1.8"
done
