#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Note that usually LoRA needs to use larger learning rate
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=/ssd/output_test_202504111304
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=0
fi
mkdir -p $OUTPUT

CUDA_VISIBLE_DEVICES=0,1;DS_SKIP_CUDA_CHECK=1 deepspeed --include localhost:1 --master_port 5524 main_pipeline.py \
 --enable_tensorboard \
 --tensorboard_path $OUTPUT \
 --output_dir $OUTPUT &> $OUTPUT/training.log
