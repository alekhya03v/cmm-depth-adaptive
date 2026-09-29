#!/bin/bash
M=clip-vit-base-patch32; DD="method.alpha_first=0.7 method.alpha_last=0.3"
for s in 0 1 2; do
  # 1) where does global strength peak?
  echo "vitb32_TA8_dac_sc1.6|$M|TA8|dacmerge|$s|method.scale_first=1.6 method.scale_last=1.6"
  echo "vitb32_TA8_dac_sc1.8|$M|TA8|dacmerge|$s|method.scale_first=1.8 method.scale_last=1.8"
  echo "vitb32_TA8_dac_sc2.0|$M|TA8|dacmerge|$s|method.scale_first=2.0 method.scale_last=2.0"
  # 2) depth-aware strength with the SAME average as global 1.4 and 1.6, plus depth-down
  echo "vitb32_TA8_dac_dd_sc1.2-1.6|$M|TA8|dacmerge|$s|$DD method.scale_first=1.2 method.scale_last=1.6"
  echo "vitb32_TA8_dac_dd_sc1.4-1.8|$M|TA8|dacmerge|$s|$DD method.scale_first=1.4 method.scale_last=1.8"
  echo "vitb32_TA8_dac_dd_sc1.4|$M|TA8|dacmerge|$s|$DD method.scale_first=1.4 method.scale_last=1.4"
done
