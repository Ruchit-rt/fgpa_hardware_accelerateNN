#!/bin/bash
#PBS -l walltime=24:00:00 
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
export PKG_CONFIG_PATH=/home/u196631/opencv_install/lib/pkgconfig:$PKG_CONFIG_PATH
export QUARTUS_ROOTDIR=/glob/development-tools/versions/intelFPGA_pro/19.2/quartus/
# export QUARTUS_ROOTDIR_OVERRIDE=/glob/development-tools/versions/intelFPGA_pro/19.2/quartus/
# source /glob/development-tools/versions/fpgasupportstack/d5005/2.0.1/inteldevstack/init_env.sh
export QSYS_ROOTDIR=/glob/development-tools/versions/intelFPGA_pro/19.2/qsys/bin
export PATH="$QUARTUS_ROOTDIR/bin:$QSYS_ROOTDIR:$PATH"
cd /home/u196631/urop/final-year-project/build
cmake ..
make fpga_emu

#  qsub -l nodes=1:stratix10:ppn=2 -d . scripts/emulate_build.sh -l walltime=5:00:00