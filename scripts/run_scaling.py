import nimare
import os
import numpy as np
import nilearn.maskers
import nibabel
import sys
from functools import wraps
from time import time
import json

from run_meta import run_ale, run_sale

# define input and output directories
INPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'input')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')

def timing(f):
    """
    Timing decorator which times
    the function and returns the duration and results
    """
    @wraps(f)
    def wrap(*args, **kw):
        start = time()
        result = f(*args, **kw)
        duration = time()-start
        return duration, result
    return wrap

@timing
def run_ale_timed(dset, mask, n_iters=10000, use_gpu=True, n_cores=1):
    np.random.seed(0)
    run_ale(
        dset, mask, n_iters=n_iters, 
        use_gpu=use_gpu, n_cores=n_cores,
    )
        
@timing
def run_sale_timed(dset, mask, n_iters=10000, use_gpu=True, n_cores=1, approach='probabilistic'):
    np.random.seed(0)
    run_sale(
        dset, mask, n_iters=n_iters,
        use_gpu=use_gpu, n_cores=n_cores, 
        approach=approach
    )

def run_scaling(analysis, n_iters, n_exp, use_gpu=False):
    """
    Run the analysis on the specified number of experiments
    and records the computing time

    Parameters
    ----------
    analysis : {'ale', 'sale', 'dsale'}
    n_iters : int
        number of iterations for the null distribution
    
    """
    jsons_dir = os.path.join(OUTPUT_DIR, 'scaling_jsons')
    os.makedirs(jsons_dir, exist_ok=True)
    # load the cerebellum dilated mask
    mask_img = nibabel.load(os.path.join(INPUT_DIR, 'maps','D2009_MNI_dilated-6mm.nii.gz'))
    mask = nilearn.maskers.NiftiMasker(mask_img)
    # load dump and create a dataset from n_exp first subset of experiments
    dump = nimare.dataset.Dataset.load(os.path.join(OUTPUT_DIR, 'data', 'BrainMap_dump_Feb2024_mask-D2009_MNI_dilated-6mm.pkl.gz'))
    dset = dump.slice(dump.ids[:n_exp])
    # run analysis and time it
    if analysis == 'ale':
        duration, _ = run_ale_timed(dset, mask, n_iters=n_iters, use_gpu=use_gpu, n_cores=1)
    elif analysis == 'dsale':
        duration, _ = run_sale_timed(dset, mask, n_iters=n_iters, use_gpu=use_gpu, n_cores=1, approach='deterministic')
    elif analysis == 'sale':
        duration, _ = run_sale_timed(dset, mask, n_iters=n_iters, use_gpu=use_gpu, n_cores=1, approach='probabilistic')
    # save the timing data to a json file
    data = {
        'n_exp': n_exp,
        'n_iters': n_iters,
        'duration': duration,
        'hardware': 'gpu' if use_gpu else 'cpu',
        'analysis': analysis
    }
    filename = f'{analysis}_{data["hardware"]}_exp-{n_exp}_iters-{n_iters}.json'
    with open(os.path.join(jsons_dir, filename), 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == '__main__':
    analysis = sys.argv[1]
    n_iters = int(sys.argv[2])
    n_exp = int(sys.argv[3])
    use_gpu = bool(int(sys.argv[4]))
    run_scaling(analysis, n_iters, n_exp, use_gpu)