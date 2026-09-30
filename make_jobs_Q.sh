#!/bin/bash
M=clip-vit-base-patch32
BEST="method.alpha_first=0.7 method.alpha_last=0.3 method.scale_first=1.2 method.scale_last=2.4"
for s in 0 1; do
  echo "vitb32_TA8_dac_best_keepcross|$M|TA8|dac3merge|$s|$BEST method.proj_mode=keepcross"
  echo "vitb32_TA8_dac_best_keepcrossh|$M|TA8|dac3merge|$s|$BEST method.proj_mode=keepcross_h"
  echo "vitb32_TA8_dac_best_lnpre|$M|TA8|dac3merge|$s|$BEST method.ln_mode=pre"
  echo "vitb32_TA8_dac_best_lnhalf|$M|TA8|dac3merge|$s|$BEST method.ln_mode=half"
done
