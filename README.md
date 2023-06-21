# `Accelerate Interpretable Deep Neural Networks for Medical Imaging`

project description.

## Setting Optional Flags

| Variable              | Description
|:---                   |:---
| `USER_HARDWARE_FLAGS` | This semicolon-separated list of flags applies only to flows that generate FPGA hardware (i.e. report, simulation, hardware). You can specify flags such as `-Xsclock` or `-Xshyper-optimized-handshaking=off`
| `USER_FLAGS`          | This semicolon-separated list of flags applies to all flows, including emulation. You can specify flags such as `-v` or define macros such as `-DYOUR_OWN_MACRO=3`
| `USER_INCLUDE_PATHS`  | This semicolon-separated list of include paths applies  to all flows, including emulation. Specify include paths relative to the `CMakeLists.txt` file, or using absolute paths in the filesystem.

## Building Options

This design uses CMake to generate a build script for GNU/make.

1. Configure the build system for the Stratix10 device family, which is the default.

   ```
   mkdir build
   cd build
   cmake ..
   ```

2. Compile the design through the generated `Makefile`.

   | Target          | Expected Time  | Output                                                                       | Description
   |:---             |:---            |:---                                                                          |:---
   | `make fpga_emu` | Seconds        | x86-64 binary                                                                | Compiles the FPGA device code to the CPU. Use the Intel® FPGA Emulation Platform for OpenCL™ software to verify your SYCL code’s functional correctness.
   | `make report`   | Minutes        | RTL + FPGA reports                                                           | Compiles the FPGA device code to RTL and generates an optimization report that describes the structures generated on the FPGA, identifies performance bottlenecks, and estimates resource utilization. This report will include the interfaces defined in your selected Board Support Package.
   | `make fpga_sim` | Minutes        | RTL + FPGA reports + x86-64 binary                                           | Compiles the FPGA device code to RTL and generates a simulation testbench. Use the Questa*-Intel® FPGA Edition simulator to verify your design.
   | `make fpga`     | Multiple Hours | Quartus Place & Route (Full accelerator) + FPGA reports + x86-64 host binary | Compiles the FPGA device code to RTL and compiles the generated RTL using Intel® Quartus® Prime. If you specified a BSP with `FPGA_DEVICE`, this will generate an FPGA image that you can run on the corresponding accelerator board.
   | `make fpga_ip_export` | Minutes  | RTL + FPGA reports                                                           | Compiles the FPGA device code to RTL that may be exported to Intel® Quartus Prime software

## Run the `fpga_template` Executable

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   ./fpga_template.fpga_emu
   ```
2. Run the sample on the FPGA simulator device.
   ```
   CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./fpga_template.fpga_sim
   ```
3. Alternatively, run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   ./fpga_template.fpga
   ```