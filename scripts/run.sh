#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling urop_model.cpp
icpx -fsycl -std=c++17 src/urop_model.cpp
if [ $? -eq 0 ]; then ./a.out; fi
