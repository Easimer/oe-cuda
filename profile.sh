#!/bin/sh

/usr/local/cuda/bin/nvprof -m achieved_occupancy $@ > /dev/null
/usr/local/cuda/bin/nvprof $@ > /dev/null
