#!/bin/bash
# Job files for the full OPCM Table 1 reproduction.
# OPCM runs use symerge keep_synergy=false = identical to OPCM + flaw diagnostics.
line() { echo "$1|$2|$3|$4|$5|$6"; }
gen() {
  S=$1; M=$2; U=$3
  [ "$U" = "TA8" ] && LAM=0.3 || LAM=0.1
  for s in 0 1 2 3 4; do line ${S}_${U}_opcm $M $U symerge $s "method.keep_synergy=false"; done
  for s in 0 1 2; do
    line ${S}_${U}_weight_average  $M $U weight_average  $s ""
    line ${S}_${U}_task_arithmetic $M $U task_arithmetic $s "method.scaling_factor=$LAM"
    line ${S}_${U}_ties_merging    $M $U ties_merging    $s "method.scaling_factor=$LAM"
  done
}
{ gen vitb32 clip-vit-base-patch32 TALL20; gen vitb32 clip-vit-base-patch32 TALL14; } > jobs_A_vitb32.txt
{ gen vitb16 clip-vit-base-patch16 TALL20; gen vitb16 clip-vit-base-patch16 TALL14; gen vitb16 clip-vit-base-patch16 TA8; } > jobs_B_vitb16.txt
{ gen vitl14 clip-vit-large-patch14 TALL20; gen vitl14 clip-vit-large-patch14 TALL14; gen vitl14 clip-vit-large-patch14 TA8; } > jobs_C_vitl14.txt
for a in 0.1 0.2 0.3 0.4 0.6 0.7 0.8 0.9 1.0; do for s in 0 1 2; do
  line vitb32_TA8_opcm_alpha$a clip-vit-base-patch32 TA8 symerge $s "method.keep_synergy=false method.alpha=$a"
done; done > jobs_D_alpha.txt
wc -l jobs_*.txt
