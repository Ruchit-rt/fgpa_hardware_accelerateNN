#!/bin/bash
#PBS -l walltime=24:00:00 
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
export PKG_CONFIG_PATH=/home/u196631/opencv_install/lib/pkgconfig:$PKG_CONFIG_PATH
export QUARTUS_ROOTDIR=/glob/development-tools/versions/intelFPGA_pro/19.2/quartus/
# export QUARTUS_ROOTDIR_OVERRIDE=/glob/development-tools/versions/intelFPGA_pro/19.2/quartus/
# source /glob/development-tools/versions/fpgasupportstack/d5005/2.0.1/inteldevstack/init_env.sh
export QSYS_ROOTDIR=/glob/development-tools/versions/intelFPGA_pro/19.2/qsys/bin
export PATH="$QUARTUS_ROOTDIR/bin:$QSYS_ROOTDIR:$PATH"
cd /home/u196631/urop/final-year-project/
icpx -v -fsycl -fintelfpga -DFPGA_HARDWARE src/model_quantised.cpp -Xshardware -o tester_2Kernel_repeat.fpga -Xstarget=intel_s10sx_pac:pac_s10_usm $(pkg-config --cflags --libs opencv4)
