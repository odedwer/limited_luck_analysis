# Limited Luck in Space

This repository contains the data and analysis code for the limited luck in space project.

## Project structure

```
.
├── data/
│   ├── processed
│   └── raw
├── figures/
│   ├── main
│   └── si
├── outputs/
├── images/
├── README.md
├── preprocess.py
├── experiment1.py
├── experiment2.py
├── spatial_gamblers_fallacy.py
├── utils.py
├── jags_model_selection.R
└──  requirements.txt
```

### _data_ directory

Contains the raw data downloaded from qualtrics (within the _raw_ directory). The data is in CSV format, where each row
is the data from a single response to the experiment.
Also contains the data after running preprocess.py (within the _processed_ directory), which removes incomplete
responses and rejects responses according to the pre-registration specifications.

### _figures_ directory

Contains the figures generated from the different scripts. Figures included in the main paper, are under the _main_
directory. Figures for the supplementary information are under the _si_ directory.

### _outputs_ directory

Contains the console outputs of each of the python scripts. The outputs contain regression tables and permutation test
results.

### _images_ directory

Contains the images used for heatmap plots, downloaded from the Qualtrics experiments.

### Project root

Contains the scripts used to perform the analysis and the python requirement file.

1. **_preprocess.py_** - processes the raw data, removing irrelevant columns and rejecting responses.
2. **_experiment1.py_** - runs the analysis and produces the plots for the first experiment.
3. **_experiment2.py_** - runs the analysis and produces the plots for the second experiment, that should be a
   reproduction of the first.
4. **_gamblers_fallacy.py_** - runs the analysis and produces the plots for the spatial gamblers fallacy experiment.
5. **_utils.py_** - constants and shared utility functions.
6. **_jags_model_selection.R_** - performs Bayesian model fitting and BIC based selection for unimodel vs bimodal
   hypothesis for the data.   