# we strongly recommend to use MPI; if you don't want to / can't use MPI, start the 
# dommand from "python" onwards
# if you use MPI, set the number of cores with the "-np" flag. Don't use more cores than
# live-points.
mpirun -np 10 python /your/path/to/montepython_2cosmos/montepython/MontePython.py run \
	# supply relative path from working directory (wd) to param-file
        -p your/path/to/input/kids450_cf.param \
        # supply relative path from wd to output folder
        -o your/path/to/output/kids450_cf/ \
        # supply relative path from wd to correctly set config-file (otherwise default.conf from MontePython will be used)
        --conf your/path/to/your_config.conf \
        # choose the PolyChord sampler (nested sampling)
	-m PC
	# for smoother contours, increase the number of live-points (default: nparam * 25)
	# --PC_nlive 1000
