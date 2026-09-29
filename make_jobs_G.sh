#!/bin/bash
M=clip-vit-base-patch32
for b in 0.3 0.5 1.0; do
  for s in 0 1 2 3 4; do
    echo "vitb32_TA8_cap_oap_b$b|$M|TA8|capmerge|$s|method.alpha_schedule=overlap method.overlap_beta=$b"
  done
done
