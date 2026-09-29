#!/bin/bash
M=clip-vit-base-patch32
for U in TALL20 TALL14; do
  for s in 0 1 2; do
    echo "vitb32_${U}_opcm|$M|$U|capmerge|$s|"
    echo "vitb32_${U}_cap_depthdown|$M|$U|capmerge|$s|method.alpha_schedule=linear method.alpha_first=0.7 method.alpha_last=0.3"
  done
done
