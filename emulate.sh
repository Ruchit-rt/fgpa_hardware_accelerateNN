#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
icpx -std=c++17 -fsycl -fintelfpga  -DFPGA_EMULATOR src/urop_model_quant.cpp -o urop_model_quant.fpga_emu