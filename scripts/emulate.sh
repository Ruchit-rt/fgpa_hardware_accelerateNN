#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
icpx -std=c++17 -fsycl -fintelfpga  -DFPGA_EMULATOR src/model_quantised.cpp -o model_quantised.fpga_emu