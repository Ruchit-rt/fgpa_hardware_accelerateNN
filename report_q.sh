#!/bin/bash
qsub_id=$(qsub -l nodes=1:fpga_runtime:ppn=2 -d . report.sh)

job_id=$(echo "$qsub_id" | cut -d '.' -f 1)

echo "Job submitted"
echo ""
qstat 

echo -ne "\nWaiting for output file "
until [ -f report.sh.o$job_id ]; do
    sleep 1
    echo -n "."
done

cat report.sh.o$job_id
cat report.sh.e$job_id
