import nimare
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nilearn.plotting
import nilearn.maskers
import nibabel
import sys
import time
from nimare.meta.cbma.ale import ALE, SCALE
from nimare.correct import FWECorrector
from functools import wraps
from time import time
import json

np.random.seed(0)
INPUT_DIR = '/data/project/cerebellum_ale/input'
OUTPUT_DIR = '/data/project/cerebellum_ale/output'
jsons_dir = os.path.join(OUTPUT_DIR, 'scaling_jsons')
os.makedirs(jsons_dir, exist_ok=True)
mask_img = nibabel.load(os.path.join(INPUT_DIR, 'maps','D2009_MNI_dilated-6mm.nii.gz'))
mask = nilearn.maskers.NiftiMasker(mask_img)

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        start = time()
        result = f(*args, **kw)
        duration = time()-start
#         print('func:%r args:[%r, %r] took: %2.4f sec' % \
#           (f.__name__, args, kw, duration))
        return duration, result
    return wrap

@timing
def run_ale(dset, mask, n_iters=10000, n_cores=1):
    np.random.seed(0)
    n_exp = len(dset.ids)
    print(f"running ALE... ({n_exp} experiments)")
    n_cores = 1
    meta = ALE(mask=mask, null_method='approximate')
    # run true (observed) ALE
    results = meta.fit(dset)
    # run multiple comparison correction
    print(f"running FWE correction...")
    corr = FWECorrector(method="montecarlo", voxel_thresh=0.001, n_iters=n_iters, n_cores=n_cores, vfwe_only=False)
    cres = corr.transform(results)
    # clean up
    del results, cres, meta, corr
        
@timing
def run_sale(dset, mask, xyz, n_iters=10000):
    np.random.seed(0)
    n_exp = len(dset.ids)
    print(f"running dSALE... ({n_exp} experiments)")
    meta = SCALE(
        xyz=xyz,
        mask=mask,
        n_iters=n_iters,
        n_cores=1,
    )
    # run SALE and its permutations
    results = meta.fit(dset)
    # clean up
    del results, meta

def run(analysis, n_iters, n_exp, use_gpu=False):
    if use_gpu:
        raise NotImplementedError
    dump = nimare.dataset.Dataset.load(os.path.join(OUTPUT_DIR, 'data', 'BrainMap_dump_Feb2024_mask-D2009_MNI_dilated-6mm.pkl.gz'))
    dset = dump.slice(dump.ids[:n_exp])
    if analysis == 'ale':
        duration, _ = run_ale(dset, mask, n_iters=n_iters)
    elif analysis == 'dsale':
        xyz = dump.coordinates[['x', 'y', 'z']].values
        duration, _ = run_sale(dset, mask, xyz, n_iters=n_iters)
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
    run(analysis, n_iters, n_exp)