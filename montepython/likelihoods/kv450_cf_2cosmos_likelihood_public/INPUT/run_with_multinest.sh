#!/bin/bash/ 
python /your/path/to/montepython_2cosmos_public/montepython/MontePython.py run \
	`# supply relative path from working directory (wd) to param-file` \
	-p your/path/to/input/kv450_cf_2cosmos.param \
	`# supply relative path from wd to output folder` \
	-o your/path/to/output/kv450_cf_2cosmos/ \
	`# supply relative path from wd to correctly set config-file (otherwise default.conf from MontePython will be used)` \
	--conf your/path/to/your_config.conf \
	`# choose the MultiNest sampler (nested sampling)` \
	-m NS \
	`# set an arbitrary but large number of steps (run should converge well before!)` \
	--NS_max_iter 10000000 \
	`# flag for using importance nested sampling (we did not use it, but it might be faster when using it!)` \
	--NS_importance_nested_sampling False \
	`# for parameter estimation use 0.8 (0.3 recommended for more accurate evidences, and we need these)` \
	--NS_sampling_efficiency 0.3 \
	`# the more live points the smoother the contours, empirical number, experiment (depends also on hardware available)` \
	--NS_n_live_points 1000 \
	`# run will finish/is converged if ln(Z_i) - ln(Z_j) <= NS_evidence_tolerance for i>j (0.1 is for evidences)` \  
	--NS_evidence_tolerance 0.1
