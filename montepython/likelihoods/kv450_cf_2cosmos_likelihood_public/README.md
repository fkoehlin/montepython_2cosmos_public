This folder contains the '2cosmos' likelihood module to be used with the KiDS+VIKING-450 (in short: KV450) correlation function measurements from [Hildebrandt et al. 2018 (arXiv:1812.06076)](http://adsabs.harvard.edu/abs/2018arXiv181206076H). This likelihood was used in their Section 7.4 for a consistency analysis following the methodology developed in [Koehlinger et al. 2019 (MNRAS, 484, 3126)](http://adsabs.harvard.edu/abs/2019MNRAS.484.3126K). 
The likelihood can only be run within this modified version of Monte Python, i.e. [montepython_2cosmos_public](https://github.com/fkoehlin/montepython_2cosmos_public) which calls two independent instances of [CLASS](https://github.com/legourg/class_public) (version >= 2.6!) and which also keeps the relevant cosmological parameter dictionaries separate, so that each subset of the mutually exclusive split of the main dataset is assigned its own copy of cosmological (and nuisance) parameters. The two subsets are still coupled through their joint covariance though to properly account for their correlations. To compare the split parameters then to the joint parameters and to compare the evidences for the Bayes factor a comparison run with the regular [kv450_cf_likelihood_public](https://github.com/fkoehlin/kv450_cf_likelihood_public) is required within a standard [MontePython](https://github.com/brinckmann/montepython_public) and [CLASS](https://github.com/lesgourg/class_public) (version >= 2.6 and including the HMcode module) setup. The required KiDS+VIKING-450 data files can be downloaded from the [KiDS science data webpage](http://kids.strw.leidenuniv.nl/sciencedata.php) and the parameter file for reproducing the split runs of [Hildebrandt et al. 2018 (arXiv:1812.06076)](http://adsabs.harvard.edu/abs/2018arXiv181206076H) is supplied in the subfolder `INPUT` within this repository.

Assuming that 'montepython_2cosmos_public' (with CLASS version >= 2.6) is set up (we recommend to use the MultiNest sampler!), please proceed as follows:

1) Set the path to the data folder (i.e. `KV450_COSMIC_SHEAR_DATA_RELEASE` from the tarball available from the [KiDS science data webpage](http://kids.strw.leidenuniv.nl/sciencedata.php') in `kv450_cf_2cosmos_likelihood_public.data` and modify parameters as you please (note that everything is set up to work with `kv450_cf_2cosmos.param`).

2) Start your runs using e.g. the `kv450_cf_2cosmos.param` supplied in the subfolder `INPUT` within this repository.

3) If you publish your results based on using this likelihood, please cite [Hildebrandt et al. 2018 (arXiv:1812.06076](http://adsabs.harvard.edu/abs/2018arXiv181206076H) and all further references for the KiDS+VIKING-450 data release (as listed on the [KiDS science data webpage](http://kids.strw.leidenuniv.nl/sciencedata.php)). Please cite also [Koehlinger et al. 2019 (MNRAS, 484, 3126)](http://adsabs.harvard.edu/abs/2019MNRAS.484.3126K) for using the '2cosmos' version of Monte Python as well as all relevant references for the standard MontePython and CLASS.

Refer to `run_with_multinest.sh` within the subfolder `INPUT` for all MultiNest-related settings that were used for the runs.

Note when you run the likelihood for the very first time, the covariance matrix from the data release (given in list format) needs to be converted into an actual NxN matrix format. This will take several minutes, but only once. The reformatted matrix will be saved to and loaded for all subsequent runs of the likelihood from the folder `FOR_MONTE_PYTHON` within the main folder `KV450_COSMIC_SHEAR_DATA_RELEASE` of the data release.

WARNING: This likelihood only produces valid results for `\Omega_k = 0`, i.e. flat cosmologies!