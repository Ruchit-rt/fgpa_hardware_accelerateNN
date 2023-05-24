emulate:
	scripts/submit_q emulate.sh

simulate:
	scripts/submit_q simulate.sh

report:
	scripts/submit_q report.sh

hardware:
	scripts/submit_q hardware.sh

run_emulate:
	scripts/submit_q run_emu.sh

run_simulate:
	scripts/submit_q run_sim.sh

run_hardware:
	scripts/submit_q run_hardware.sh

clean: 
	rm jobs/*.sh.* > /dev/null 2>&1
