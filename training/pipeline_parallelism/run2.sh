#!/bin/bash

deepspeed train.py --deepspeed_config=ds_config.json -p 3 --steps=200
