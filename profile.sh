#!/bin/sh

/usr/local/cuda/bin/nvprof  --metrics achieved_occupancy $@
