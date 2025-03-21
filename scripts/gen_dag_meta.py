import sys
import os
import nimare.dataset

import utils

# define input and output directories
INPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'input')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')

def generate_dagman_file(analysis, source, subsampling, subsample_subbds=False):
    """
    Generates a .dagman file for running the meta-analysis jobs using HTCondor

    Parameters
    ----------
    analysis : {'ale', 'sale', 'dsale'}
    source : {'BrainMap', 'Neurosynth"}
    subsampling : float
        - 0: no subsampling
        - [0, 1): subsample propoertion (p)
        - >= 1: subsample size (N)
    subsample_subbds : bool
        whether to include sub-bds in the subsampling
    """
    # make dag directory
    dag_dir = os.path.join(os.path.dirname(__file__), 'dag')
    os.makedirs(dag_dir, exist_ok=True)
    # determine dagman file name
    prefix = f'run_{analysis}_{source}'
    if subsampling >= 1:
        prefix += f'_subsampling_N-{int(subsampling)}'
    elif (subsampling < 1) & (subsampling > 0):
        prefix += f'_subsampling_p-{subsampling}'
    if subsample_subbds:
        prefix += '_subbds'
    submit_file = os.path.join(os.path.dirname(__file__), 'run_meta.submit')
    dag_path = os.path.join(dag_dir, prefix+'.dag')
    f = open(dag_path, 'w')
    if source == 'BrainMap':
        # list all subbds from the dump
        dump = nimare.dataset.Dataset.load(os.path.join(OUTPUT_DIR, 'data', 'BrainMap_dump_Feb2024.pkl.gz'))
        # convert the behavioral domains to lists
        dump.metadata['behavioral_domain'] = dump.metadata['behavioral_domain'].map(lambda s: s.split(";"))
        # get a list of all bds and subbds
        all_bds = ['Action', 'Cognition', 'Emotion', 'Interoception', 'Perception']
        all_subbds = sorted(list(set(dump.metadata['behavioral_domain'].sum())))
    elif source == 'Neurosynth':
        all_bds = []
        all_subbds = utils.neurosynth_terms
    # set n_subsamples and subsample_size
    n_subsamples = 0
    subsample_size = 0
    if subsampling > 0:
        # set n and size of subsamples
        n_subsamples = 50
        if subsampling >= 1:
            subsample_size = int(subsampling)
        elif subsampling < 1:
            subsample_size = subsampling
        if subsample_subbds:
            domains = all_subbds
        else:
            domains = all_bds
    else:
        domains = all_bds + all_subbds
    for i, domain in enumerate(domains):
        # replace illegal characters in subbd names
        # these will be replaced back in run_meta.py
        subbd = domain.replace(' ', '_space_').replace('/', '_slash_')
        # define job variables
        f.write(f'JOB job_{i} {submit_file}\n')
        f.write(f'VARS job_{i} analysis="{analysis}" source="{source}" subbd="{subbd}" n_subsamples="{n_subsamples}" subsample_size="{subsample_size}"\n\n')
    maxjobs = 5 # number of ongoing jobs at a time
    f.write(f'\nCATEGORY ALL_NODES {prefix}')
    f.write(f'\nMAXJOBS {prefix} {maxjobs}\n\n') # keep three GPUs free


if __name__ == '__main__':
    analysis = sys.argv[1]
    source = sys.argv[2]
    subsampling = float(sys.argv[3])
    try:
        subsample_subbds = bool(int(sys.argv[4]))
    except IndexError:
        subsample_subbds = False
    generate_dagman_file(analysis, source, subsampling, subsample_subbds)
