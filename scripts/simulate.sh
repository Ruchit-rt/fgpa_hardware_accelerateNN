#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
icpx -fsycl -fintelfpga -DFPGA_SIMULATOR src/model_quantised.cpp -Xssimulation -Xsghdl -o model_quantised.a
