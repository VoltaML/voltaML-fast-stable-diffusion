#!/bin/bash
LD_PRELOAD=lib/trt/lib/libnvinfer_plugin.so python3 main.py --in-container --log-level=${LOG_LEVEL} --host