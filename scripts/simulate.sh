source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
make fpga_sim
if [ $? -eq 0 ]; then CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./model_quantised.fpga_sim; fi