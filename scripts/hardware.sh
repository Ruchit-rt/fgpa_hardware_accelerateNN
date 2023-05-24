#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
icpx -fsycl -fintelfpga -DFPGA_HARDWARE src/model_quantised.cpp -Xshardware -Xstarget=Stratix10 -o model_quantised.fpga
