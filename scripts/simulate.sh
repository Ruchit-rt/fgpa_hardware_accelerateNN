source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
icpx -fsycl -fintelfpga -DFPGA_SIMULATOR src/model_quantised.cpp -Xssimulation -Xstarget=Stratix10 -Xsghdl -o model_quantised.a
if [ $? -eq 0 ]; then CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./model_quantised.fpga_sim; fi