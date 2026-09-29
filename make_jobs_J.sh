#!/bin/bash
M=clip-vit-base-patch32; DD="method.alpha_first=0.7 method.alpha_last=0.3"
for s in 0 1 2; do
  echo "vitb32_TA8_dac_sc1.2|$M|TA8|dacmerge|$s|method.scale_first=1.2 method.scale_last=1.2"
  echo "vitb32_TA8_dac_sc1.4|$M|TA8|dacmerge|$s|method.scale_first=1.4 method.scale_last=1.4"
  echo "vitb32_TA8_dac_sc1.0-1.2|$M|TA8|dacmerge|$s|method.scale_first=1.0 method.scale_last=1.2"
  echo "vitb32_TA8_dac_dd_sc0.9-1.1|$M|TA8|dacmerge|$s|$DD method.scale_first=0.9 method.scale_last=1.1"
  echo "vitb32_TA8_dac_dd_sc1.0-1.2|$M|TA8|dacmerge|$s|$DD method.scale_first=1.0 method.scale_last=1.2"
  echo "vitb32_TA8_dac_dd_sc0.8-1.2|$M|TA8|dacmerge|$s|$DD method.scale_first=0.8 method.scale_last=1.2"
done
