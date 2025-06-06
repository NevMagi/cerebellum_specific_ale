# Cerebellum-Specific ALE (C-SALE)

This repository contains code and data associated with the paper ['A bias-accounting meta-analytic approach refines and expands
the cerebellar behavioral topography'](https://doi.org/10.1101/2024.10.31.621398) by Magielse, Manoli, Eickhoff, Fox, Saberi*, and Valk*.

## Method figure
![alt text](./output/Figure8.png)

## Short Description
In this study we adapted the [Activation Likelihood Estimation method for coordinate-based meta-analysis](https://doi.org/10.1016/j.neuroimage.2011.09.017) to account for unequal distributions of reported effects. We describe how in the entire brain, but especially the cerebellum, foci distributions are highly skewed. The overrepresentation of superior cerebellar foci - likely related to historical neglect of the cerebellum in neuroimaging studies - results in inaccurate locations of cerebellar convergence across task domains. Our new method, Cerebellum-Specific ALE (C-SALE) much improves specificity of convergence in the cerebellum in both BrainMap and NeuroSynth. We extensively characterize these maps, through repeated subsampling and correspondence to existing cerebellar parcellations. We then use this new debiased framework to perform whole-brain meta-analytic connectivty modelling (MACM), showing brain-wide cerebellar coactivation networks and illustrating how the new method can be translated to any volumetric brain region-of-interest. Besides providing our full code (except the raw BrainMap data, see **Additional Requirements**) we provide a graphical-processing unit implementation of ALE that can greatly speed-up analyses at: [amnsbr/nimare-gpu](https://github.com/amnsbr/nimare-gpu).

## Repository Structure
- `scripts/`: Includes the scripts used to run the analyses and produce full output of this study.
    - `Figures/`: Contains Jupyter Notebooks used to generate the figures for the paper. Figures are numbered as in the preprint.
    - `run_meta.py`: Runs the ALE or C-SALE meta-analyses. Usage: `python run_meta.py --help`.
      We ran the meta-analyses using HTCondor on our (INM-7 Forschungszentrum Jülich) cluster by first creating DAGman files using `gen_dag_meta.py` which
      submit jobs based on `run_meta.submit`.
    - `run_macm.py`: Runs the MACM analysis with significant clusters of input thresholded map. Usage: `python run_macm.py --help`.
      We ran the MACM meta-analyses using HTCondor on our cluster by first creating DAGman files using `gen_dag_macm.py` which
      submit jobs based on `run_macm.submit`.
    - `run_variogram.py`: Calculates variogram-based SA-preserving surrogates of the input unthresholded map. Usage: `python run_variogram.py --help`.
      We create the variogram surrogates using HTCondor on our cluster by first creating DAGman files using `gen_dag_variogram.py` which
      submit jobs based on `run_variogram.submit`.
    - `run_scaling.py`: Runs scaling analyses of compute time on CPU and GPU. Usage: `python run_scaling.py <analysis> <n_iters> <n_exp> <use_gpu>`.
      We ran these using HTCondor on our cluster by first creating DAGman files using `gen_dag_scaling.py` which
      submit jobs based on `run_scaling_cpu.submit` or `run_scaling_gpu.submit`.
    - `utils.py`: Utility functions used to run the analyses and create the figures
- `tools/`: Includes SUITPy. This toolkit is a dependency of most scripts.
- `input/`: Necessary input to perform our study. Note that this excludes raw BrainMap data.
- `output/`: Full output created in this study.
    - `data/`: Data derived from BrainMap used in the meta-analyses.
    - `SALE/`: Results of the C-SALE meta-analyses.
    - `ALE/`: Results of the ALE meta-analyses.
    - `exp_stats.csv`: Experiment statistics for each BrainMap behavioral domain in C-SALE analyses.
    - `exp_stats-NS.csv`: Experiment statistics for each NeuroSynth term in C-SALE analyses.
    - `graphical_abstract.png`: A graphical abstract for the repository README.
    - `macm_exp_stats.csv`: Experiment statistics for each BrainMap behavioral domain in the MACM analyses.

## System Requirements
- Software dependencies: Find all Python dependencies in requirements.txt (see also **Installation Guide**, point 3.):
- Operating systems: Our computational cluster operates on Debian 12.
- Non-standard hardware: We used Nvidia GeForce GTX 1080 Ti GPUs for our meta-analyses. We additionally used Nvidia Tesla P100 GPUs (via Kaggle) for the scaling analyses given that the 1080 Ti GPUs were not available at the time of the analysis, but will redo the scaling with the 1080 Ti GPUs in the next version of the manuscript.

## Additional Requirements
In addition to the Python dependencies specified in `requirements.txt` and the `tools` directory, the following are required for the scripts to run:
- The BrainMap dataset is expected to be located at `output/data/BrainMap_dump_Feb2024.pkl.gz`. Due to access restrictions, this file and its derivative datasets used in the meta-analyses are not shared in the repository but can be provided upon request and after completion of a data usage agreement. Note that the BrainMap database is also searchable using [Sleuth](https://www.brainmap.org/sleuth/).
- The NeuroSynth dataset is expected to be located at `output/data/NeuroSynth_dump.pkl.gz`.
- The environment variables `$PROJECT_DIR` must be defined and point to the project's root directory.

## Installation Guide
### Instructions
1. Clone the repository:
```bash
git clone https://github.com/NevMagi/cerebellum_specific_ale.git
cd cerebellum_specific_ale
```
2. Set up Python virtual environment (optional but strongly recommended):
```bash
python3 -m venv env
source env/bin/activate
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Duration
Setup should take roughly 10 minutes, depending on your internet connection.

## Demo
We unfortunately cannot provide a full demo of our main analyses, as we are not able to share the full input data. However, if the user follows the installation guide, figure notebooks can be used to reproduce our results. We provide the necessary derivatives/ intermediate data to produce our figures. Furthermore, we show our full code for obtaining these derivatives from the input data (not executable unless the user obtains data through Collaboration Usage Agreement (brainmap.org/collaborations.html)).

## Instructions for Use

### How to use the software on (y)our data
C-SALE analyses can be run on (y)our data using the run scripts in our GitHub repository. The main analyses are performed by `run_meta.py` (C-SALE) and `run_macm.py` (MACM) (including `run_meta.submit` and `run_macm.submit` to submit jobs to the computational cluster using HTCondor). Please see the repository's README.md at the root to find all usage. Each python script has more specific instructions on usage.

## Support
If you have any questions, feel free to contact Neville Magielse (neville.magielse\[at\]gmail.com) or Amin Saberi (amnsbr\[at\]gmail.com).
