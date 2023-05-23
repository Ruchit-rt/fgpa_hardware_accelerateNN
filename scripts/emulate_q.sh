#!/bin/bash
qsub_id=$(qsub -l nodes=1:fpga_compile:ppn=2 -d . emulate.sh)

job_id=$(echo "$qsub_id" | cut -d '.' -f 1)

echo "Job submitted"
echo ""
qstat 

echo -ne "\nWaiting for output file "
until [ -f emulate.sh.o$job_id ]; do
    sleep 1
    echo -n "."
done

cat emulate.sh.o$job_id
cat emulate.sh.e$job_id
