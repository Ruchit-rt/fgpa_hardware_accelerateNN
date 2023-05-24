# FPGA hardware image:
icpx -fsycl -fintelfpga -DFPGA_HARDWARE src/model_quantised.cpp -Xshardware -Xstarget=Stratix10 -o model_quantised.fpga