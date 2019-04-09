#!/usr/bin/env bash
exp_dir=$1
nretries=$2

echo $exp_dir
echo $nretries
echo $exp_dir/outputs
echo $exp_dir/outputs_retry$nretries

cp -r $exp_dir/outputs $exp_dir/outputs_retry$nretries
