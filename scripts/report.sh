#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
# FPGA early image (with optimization report): FPGA device family
icpx -fsycl -fintelfpga -Xshardware -fsycl-link=early src/model_quantised.cpp -o model_quantised.a

icpx -fsycl -fintelfpga -Xshardware -fsycl-link=early src/model_quantised_cpu.cpp -o model_quantised_cpu.a
