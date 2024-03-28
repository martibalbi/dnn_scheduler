#!/bin/sh
DEVICE="MAX78000"
TARGET="sdk/Examples/$DEVICE/CNN"
COMMON_ARGS="--device $DEVICE --timer 0 --display-checkpoint --verbose"

python ai8xize.py --test-dir $TARGET --prefix dnn_scheduler --checkpoint-file trained/dnn-scheduler-qat8-q.pth.tar --config-file networks/dnn_scheduler.yaml $COMMON_ARGS "$@"
