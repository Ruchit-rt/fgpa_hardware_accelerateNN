#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./model_quantised.fpga_sim
