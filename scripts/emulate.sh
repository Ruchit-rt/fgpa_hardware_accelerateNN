#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
icpx -fsycl -fintelfpga -DFPGA_EMULATOR src/model_quantised.cpp -o model_quantised.fpga_emu
icpx -fsycl -fintelfpga -DFPGA_EMULATOR src/model_quantised_new.cpp -o model_quantised_new.fpga_emu $(pkg-config --cflags --libs opencv4)
icpx -fsycl -fintelfpga -DFPGA_EMULATOR src/model_quantised_stream.cpp -o model_quantised_stream.fpga_emu $(pkg-config --cflags --libs opencv4)
icpx -fsycl -fintelfpga -DFPGA_EMULATOR src/model_unoptimised.cpp -o model_unoptimised.fpga_emu