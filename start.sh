#!/bin/bash
LD_PRELOAD=${PLUGIN_LIBS} python3 main.py --in-container --log-level=${LOG_LEVEL} --host