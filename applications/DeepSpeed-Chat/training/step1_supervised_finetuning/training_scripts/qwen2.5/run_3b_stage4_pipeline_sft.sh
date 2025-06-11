#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Note that usually LoRA needs to use larger learning rate
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=/ssd2/output_test_20250528
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=0
fi
mkdir -p $OUTPUT

NCCL_DEBUG=WARN;CUDA_DEVICE_ORDER=PCI_BUS_ID;CUDA_VISIBLE_DEVICES=1,2,3,4;DS_SKIP_CUDA_CHECK=1 deepspeed   main_pipeline_grpo.py \
 --enable_tensorboard \
 --tensorboard_path $OUTPUT \
 --output_dir $OUTPUT &> $OUTPUT/training.log
