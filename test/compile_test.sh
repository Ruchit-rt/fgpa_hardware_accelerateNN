#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh #> /dev/null 2>&1
source /data/intel_fpga/devcloudLoginToolSetup.sh
tools_setup -t QP 20.1

/glob/development-tools/versions/oneapi/2023.1.2/oneapi/compiler/2023.1.0/linux/bin/icpx -fsycl -fintelfpga -Wall -qactypes -DFPGA_HARDWARE -c test/model_test.cpp -o model_test.cpp.o
/glob/development-tools/versions/oneapi/2023.1.2/oneapi/compiler/2023.1.0/linux/bin/icpx -fsycl -fintelfpga -qactypes -Xshardware -Xstarget=intel_s10sx_pac:pac_s10_usm -reuse-exe=/home/u178815/final-year-project/model_test.fpga -o model_test.fpga model_test.cpp.o
