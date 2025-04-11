#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Note that usually LoRA needs to use larger learning rate
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output_test_202503291353
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=0
fi
mkdir -p $OUTPUT

#deepspeed --num_gpus 3 main.py --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
#   --gradient_accumulation_steps 1 --lora_dim 16 --zero_stage $ZERO_STAGE \
#   --enable_tensorboard \
#   --tensorboard_path $OUTPUT \
#   --deepspeed --output_dir $OUTPUT &> $OUTPUT/training.lo
deepspeed --num_gpus 3 main.py  --enable_tensorboard --tensorboard_path $OUTPUT --deepspeed --output_dir $OUTPUT &> $OUTPUT/training.log
