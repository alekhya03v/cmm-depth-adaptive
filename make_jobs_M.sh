#!/bin/bash
M=clip-vit-base-patch32; DD="method.alpha_first=0.7 method.alpha_last=0.3"
echo "vitb32_TA8_bud_global_sanity|$M|TA8|dacbudget|0|method.budget=global"
for s in 0 1 2; do
  echo "vitb32_TA8_bud_layer|$M|TA8|dacbudget|$s|method.budget=layer"
  echo "vitb32_TA8_bud_layer_dd|$M|TA8|dacbudget|$s|method.budget=layer $DD"
  echo "vitb32_TA8_bud_param_dd|$M|TA8|dacbudget|$s|method.budget=param $DD"
  echo "vitb32_TA8_bud_layer_dd_sc1.2-1.6|$M|TA8|dacbudget|$s|method.budget=layer $DD method.scale_first=1.2 method.scale_last=1.6"
done
