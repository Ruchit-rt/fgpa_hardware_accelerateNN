#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
icpx -fsycl src/model_quantised_gpu.cpp -o model_quantised_gpu.a
