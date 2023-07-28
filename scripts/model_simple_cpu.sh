#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
icpx -fsycl src/model.cpp -o model_from_file.cpu -DCPU -I src/include $(pkg-config --cflags --libs opencv4)

