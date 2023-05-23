#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
# FPGA early image (with optimization report): FPGA device family
icpx -fsycl -fintelfpga -Xshardware -fsycl-link=early -Xstarget=Stratix10 -DFPGA_HARDWARE src/model_quantised.cpp -o model_quantised.a

