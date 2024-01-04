import nimare
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nilearn.plotting, nilearn.maskers
import nibabel
from pprint import pprint
import sys
from nimare.meta.cbma.ale import ALE
from nimare.correct import FWECorrector


INPUT_DIR = '/data/project/cerebellum_ale/input/'
OUTPUT_DIR = '/data/project/cerebellum_ale/output/'
SUBSAMPLE_SIZE = 20

def run_ale(filename, null_method='approximate', n_iters=5000, n_cores=4, 
        subsample_size=None, subsample_idx=None, n_subsamples=100):
    print("loading data...")
    # convert sleuth to nimare dset
    dset = nimare.io.convert_sleuth_to_dataset(os.path.join(INPUT_DIR, "sleuth", f'{filename}.txt'), target="mni152_2mm")
    if subsample_size is not None:
        print("running subsamples...")
        if subsample_idx is None:
            for i in range(n_subsamples):
                print(f"subsample {i}")
                run_ale(filename, null_method, n_iters, n_cores, subsample_size, i)
            return
        else:
            subsample = np.random.choice(dset.ids, subsample_size, replace=False)
            dset = dset.slice(subsample)
    # load mask
    cerebellum_mask_2mm = nibabel.load(os.path.join(INPUT_DIR, 'maps', 'cerebellumMask.nii.gz'))
    # create output folder
    out_dir = os.path.join(OUTPUT_DIR, 'ale', filename)
    if subsample_size:
        out_dir = os.path.join(out_dir, 'subsamples', str(subsample_idx))
    os.makedirs(out_dir, exist_ok=True)
    dset.save(os.path.join(out_dir, 'dset.pkl.gz'))
    # run masked ALE
    print("running ale...")
    meta = ALE(mask=cerebellum_mask_2mm, null_method=null_method)
    results = meta.fit(dset)
    # run multiple comparison correction
    print("running cFWE...")
    corr = FWECorrector(method="montecarlo", voxel_thresh=0.001, n_iters=n_iters, n_cores=n_cores)
    cres = corr.transform(results)
    # save results
    print("saving results...")
    results.save(os.path.join(out_dir, 'uncorr.pkl.gz'))
    results.save_maps(out_dir, 'uncorr')
    results.save_tables(out_dir, 'uncorr')
    cres.save(os.path.join(out_dir, 'cFWE.pkl.gz'))
    cres.save_maps(out_dir, 'cFWE')
    cres.save_tables(out_dir, 'cFWE')

if __name__ == '__main__':
    filename = sys.argv[1]
    if len(sys.argv) > 2:
        n_subsamples = int(sys.argv[2])
        if len(sys.argv) == 4:
            subsample_size = int(sys.argv[3])
        else:
            subsample_size = SUBSAMPLE_SIZE
        run_ale(filename, subsample_size=subsample_size, n_subsamples=n_subsamples)
    else:
        run_ale(filename)