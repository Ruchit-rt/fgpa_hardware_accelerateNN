#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
icpx -fsycl src/model_quantised_cpu.cpp -o model_quantised_cpu.cpu -DCPU -I src/include
