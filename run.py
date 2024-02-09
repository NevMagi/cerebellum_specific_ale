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
from nimare.meta.cbma.ale import ALE, SCALE
from nimare_gpu.ale import DeviceALE, DeviceSCALE
from nimare.correct import FWECorrector
import argparse
from numba import cuda
import json

from utils import filter_coords_to_mask, get_null_xyz, cluster_extent_correction

if 'PYTHONUTF8' not in os.environ:
    print("Rerun after `export PYTHONUTF8=1`")


INPUT_DIR = '/data/project/cerebellum_ale/input'
OUTPUT_DIR = '/data/project/cerebellum_ale/output'
SUBSAMPLE_SIZE = {
    'BD': 50,
    'SubBD': 20,
}
MIN_EXPERIMENTS = 15 # minimum number of experiments to run ALE
MASK_NAME = 'D2009_MNI_dilated-6mm' # created via utils.create_Driedrischen2009_mask('MNI')
PROB_MAP_PATH = '/data/project/cerebellum_ale/output/BrainMap_dump_Feb2024_kernels_sum.nii.gz'

def prepare_data(bd_prefix, pure=False, out_path=None):
    dump = nimare.dataset.Dataset.load(f'/data/project/cerebellum_ale/output/data/BrainMap_dump_Feb2024_mask-{MASK_NAME}.pkl.gz')
    all_bds = sorted(list(set(dump.metadata['behavioral_domain'].sum())))
    selected_bds = set(filter(lambda s: s.startswith(bd_prefix), all_bds))
    if pure:
        exp_mask = dump.metadata['behavioral_domain'].map(lambda c: len(set(c).difference(selected_bds))==0)
    else:
        exp_mask = dump.metadata['behavioral_domain'].map(lambda c: len(set(c).intersection(selected_bds))>0)
    selected_ids = dump.metadata.loc[exp_mask, 'id'].values
    dset = dump.slice(selected_ids)
    if out_path is not None:
        dset.save(out_path)
    return dset

def run_ale(dset, mask, out_dir, n_iters=10000, use_gpu=True, n_cores=4):
    print(f"running ALE...({len(dset.ids)} experiments)")
    if use_gpu:
        meta = DeviceALE(mask=mask, null_method='approximate')
        n_cores = 1
    else:
        meta = ALE(mask=mask, null_method='approximate')
    # run true (observed) ALE
    results = meta.fit(dset)
    # run multiple comparison correction
    print("running FWE correction...")
    corr = FWECorrector(method="montecarlo", voxel_thresh=0.001, n_iters=n_iters, n_cores=n_cores, vfwe_only=False)
    cres = corr.transform(results)
    # save results
    print("saving results...")
    results.save(os.path.join(out_dir, 'uncorr.pkl.gz'))
    results.save_maps(out_dir, 'uncorr')
    results.save_tables(out_dir, 'uncorr')
    cres.save(os.path.join(out_dir, 'cFWE.pkl.gz'))
    cres.save_maps(out_dir, 'cFWE')
    cres.save_tables(out_dir, 'cFWE')

def run_sale(dset, mask, out_dir, n_iters=10000, use_gpu=True, n_cores=4,
             height_thr=0.001, k=50, 
             sigma_scale=1.0, debug=False, use_mmap=False):
    print(f"running probabilistic SALE...({len(dset.ids)} experiments)")
    if use_gpu:
        meta = DeviceSCALE(
            'probabilistic',
            prob_map=nibabel.load(PROB_MAP_PATH),
            xyz=None,
            mask=mask, 
            n_iters=n_iters,
            nbits=64,
            keep_perm_nulls=debug,
            sigma_scale=sigma_scale,
            use_mmap=use_mmap,
        )
    else:
        raise NotImplementedError("CPU version of probabilistic SALE is not implemented")
    # run SALE and its permutations
    results = meta.fit(dset)
    # save uncorrected results
    print("saving results...")
    results.save(os.path.join(out_dir, 'uncorr.pkl.gz'))
    results.save_maps(out_dir, 'uncorr')
    results.save_tables(out_dir, 'uncorr')
    # run cluster-extent correction
    cres = cluster_extent_correction(results, height_thr=height_thr, k=k)
    # save corrected results
    for map_name, img in cres.items():
        img.to_filename(os.path.join(out_dir, f'corr_cluster_k-{k}_{map_name}.nii.gz'))

def run(analysis, bd, subbd, n_iters=10000, use_gpu=True, n_cores=4, 
            subsample_size=None, subsample_idx=None, n_subsamples=100):
    """
    Main function that runs ALE/SALE after some preparation
    """
    print(f"running {analysis} for {bd}", end=" ")
    print("on gpu" if use_gpu else f"on cpu ({n_cores} cores)")
    print(f"BD: {bd}")
    print(f"SubBD: {subbd}")
    # prepare dset
    dset_path = os.path.join(OUTPUT_DIR, 'data', bd, subbd, 'dset.pkl.gz')
    try:
        dset = nimare.dataset.Dataset.load(dset_path)
    except FileNotFoundError:
        print("Prepared data not found, doing it now...")
        os.makedirs(os.path.dirname(dset_path), exist_ok=True)
        dset = prepare_data(subbd, out_path=dset_path)
    if len(dset.ids) < MIN_EXPERIMENTS:
        print(f"Skipping {bd} {subbd} due to insufficient experiments")
        return
    # create output folder
    out_dir = os.path.join(OUTPUT_DIR, analysis, bd, subbd)
    if subsample_size:
        out_dir = os.path.join(out_dir, 'subsamples', str(subsample_idx))
    os.makedirs(out_dir, exist_ok=True)
    # load mask
    mask_img = nibabel.load(os.path.join(INPUT_DIR, 'maps', f'{MASK_NAME}.nii.gz'))
    mask = nilearn.maskers.NiftiMasker(mask_img)
    # subsample if indicated
    if subsample_size is not None:
        # run subsamples
        if subsample_idx is None:
            # indicating that this is the parent run
            for i in range(n_subsamples):
                run(analysis=analysis, bd=bd, subbd=subbd, n_iters=n_iters, 
                    use_gpu=use_gpu, n_cores=n_cores, subsample_size=subsample_size, 
                    subsample_idx=i)
            return
        else:
            # slice the dataset to subsample and continue
            np.random.seed(1234+subsample_idx) # reproducable but variable seeds per subsample
            subsample = np.random.choice(dset.ids, subsample_size, replace=False)
            dset = dset.slice(subsample)
            dset.save(os.path.join(out_dir, 'dset.pkl.gz'))
    print(f"subsample {subsample_idx}" if (subsample_idx is not None) else "full sample")
    # run analysis
    if analysis == 'ALE':
        run_ale(dset, mask, out_dir, n_iters=n_iters, use_gpu=use_gpu, n_cores=n_cores)
    elif analysis == 'SALE':
        run_sale(dset, mask, out_dir, n_iters=n_iters, use_gpu=use_gpu, n_cores=n_cores)
    # save config
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump({
            'analysis': analysis,
            'n_iters': n_iters,
            'use_gpu': use_gpu,
            'n_cores': n_cores,
            'subsample_size': subsample_size,
            'subsample_idx': subsample_idx,
            'n_subsamples': n_subsamples,
        }, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('analysis', type=str, help='analysis to run', choices=['ALE', 'SALE'])
    parser.add_argument('bd', type=str, help='behavioural domain to run')
    parser.add_argument('subbd', type=str, help='sub-behavioural domain to run')
    parser.add_argument('-n_subsamples', type=int, default=0, help='number of subsamples to run'
                                                                   ' (0 indicates no subsampling)')        
    parser.add_argument('-subsample_size', type=int, default=None, help='number of experiments to subsample')
    parser.add_argument('-n_iters', type=int, default=10000, help='number of null permutations')
    # parser.add_argument('-use_gpu', type=int, help='use gpu for ALE', default=True)
    parser.add_argument('-n_cores', type=int, help='number of CPU cores', default=4)

    args = parser.parse_args()

    # get default subsample size depending on bd vs subbd
    if (args.subsample_size is None) and (args.n_subsamples > 0):
        args.subsample_size = SUBSAMPLE_SIZE['BD' if args.subbd == args.bd else 'SubBD']

    # disable subsampling if subsample size is 0
    if args.subsample_size == 0:
        args.subsample_size = None

    # check for GPU
    try:
        cuda.select_device(0)
    except cuda.cudadrv.error.CudaSupportError:
        print("No GPU found, using CPU")
        use_gpu = False
    else:
        print("GPU found, using GPU")
        use_gpu = True

    run(analysis=args.analysis, bd=args.bd, subbd=args.subbd, 
        n_iters=args.n_iters, use_gpu=use_gpu, n_cores=args.n_cores, 
        subsample_size=args.subsample_size, n_subsamples=args.n_subsamples)