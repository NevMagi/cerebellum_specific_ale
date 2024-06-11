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
import glob

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
             sigma_scale=1.0, debug=False, use_mmap=False,
             deterministic=False):
    if use_gpu:
        if deterministic:
            approach = 'deterministic'
            prob_map = None
            xyz = get_null_xyz(f'/data/project/cerebellum_ale/output/data/BrainMap_dump_Feb2024_mask-{MASK_NAME}.pkl.gz',
                                os.path.join(INPUT_DIR, 'maps', f'{MASK_NAME}.nii.gz'), 
                                unique=False)
        else:
            approach = 'probabilistic'
            prob_map = nibabel.load(PROB_MAP_PATH)
            xyz = None
        print(f"running {approach} SALE...({len(dset.ids)} experiments)")
        meta = DeviceSCALE(
            approach,
            prob_map=prob_map,
            xyz=xyz,
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
    # TODO: some parts of results is saved on GPU memory
    # and this will make loading them on CPU nodes not possible
    # therefore either remove them or transfer to CPU memory
    # before saving
    results.save(os.path.join(out_dir, 'uncorr.pkl.gz'))
    results.save_maps(out_dir, 'uncorr')
    results.save_tables(out_dir, 'uncorr')
    # run cluster-extent correction
    cres = cluster_extent_correction(results, height_thr=height_thr, k=k)
    # save corrected results
    for map_name, img in cres.items():
        img.to_filename(os.path.join(out_dir, f'corr_cluster_k-{k}_{map_name}.nii.gz'))

def sale_k_cluster(k=50, height=0.001, mask_name='D2009_MNI'):
    """
    Sweeps through all SALE results and applies cluster extent correction
    after applying the proper cerebellar mask
    """
    mask_img = nibabel.load(f'{INPUT_DIR}/maps/{mask_name}.nii.gz')
    uncorr_paths = glob.glob('/data/project/cerebellum_ale/output/SALE/*/*/uncorr.pkl.gz') \
        + glob.glob('/data/project/cerebellum_ale/output/SALE/*/*/subsamples*/*/uncorr.pkl.gz')
    for uncorr_path in sorted(uncorr_paths):
        clustered_path = uncorr_path.replace('uncorr.pkl.gz', f'corr_cluster_h-{str(height)[2:]}_k-{k}_mask-{mask_name}_z.nii.gz')
        if os.path.exists(clustered_path):
            continue
        print(uncorr_path)
        results = nimare.results.MetaResult.load(uncorr_path)
        z_orig = results.get_map('z')
        logp_orig = results.get_map('logp')
        # mask to the proper mask
        z_masked = nilearn.image.math_img("m * img", img=z_orig, m=mask_img)
        logp_masked = nilearn.image.math_img("m * img", img=logp_orig, m=mask_img)
        # cluster extent correction
        cres = cluster_extent_correction({'z': z_masked, 'logp': logp_masked}, k=k, height_thr=height)
        # save corrected results
        for map_name, img in cres.items():
            img.to_filename(uncorr_path.replace('uncorr.pkl.gz', f'corr_cluster_h-{str(height)[2:]}_k-{k}_mask-{mask_name}_{map_name}.nii.gz'))

def run_macm(zmap_path, n_iters=10000, use_gpu=True, n_cores=4):
    """
    Run MACM from cerebellar seed clusters in thresholded zmap 
    to the whole brain
    """
    # define seed clusters
    zmap = nibabel.load(zmap_path)
    seed = nilearn.image.binarize_img(zmap)
    # load masked dump and the whole-brain grey10 mask
    dump = nimare.dataset.Dataset.load(os.path.join(OUTPUT_DIR, 'data', 'BrainMap_dump_Feb2024_mask-Grey10.pkl.gz'))
    mask_name = 'Grey10'
    mask_img = nibabel.load(f'{INPUT_DIR}/maps/{mask_name}.nii.gz')
    # filter dump to experiments including a focus in the mask
    seed_dset = dump.slice(dump.get_studies_by_mask(seed))
    # set output path
    out_dir = zmap_path.replace('.nii.gz', '_macm')
    os.makedirs(out_dir, exist_ok=True)
    # run pSALE
    run_sale(seed_dset, mask_img, out_dir,
             n_iters=n_iters, use_gpu=use_gpu, n_cores=n_cores,
             deterministic=False)



def run(analysis, bd, subbd, n_iters=10000, use_gpu=True, n_cores=4, 
            subsample_size=None, subsample_idx=None, n_subsamples=100):
    """
    Main function that runs ALE/SALE after some preparation
    """
    subbd = subbd.replace('_space_', ' ').replace('_slash_', '/')
    print(f"running {analysis} for {bd}", end=" ")
    print("on gpu" if use_gpu else f"on cpu ({n_cores} cores)")
    print(f"BD: {bd}")
    print(f"SubBD: {subbd}")
    # prepare dset
    subbd_clean = subbd.replace(' ', '').replace('/', '') # used for subfolder names
    dset_path = os.path.join(OUTPUT_DIR, 'data', bd, subbd_clean, 'dset.pkl.gz')
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
    out_dir = os.path.join(OUTPUT_DIR, analysis, bd, subbd_clean)
    if (subsample_size is not None) & (subsample_idx is not None):
        if subsample_size >= 1:
            out_dir = os.path.join(out_dir, f'subsamples_n-{int(subsample_size)}', str(subsample_idx))
        else:
            out_dir = os.path.join(out_dir, f'subsamples_p-{subsample_size}'.replace('.',''), str(subsample_idx))
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
            # determine subsample size
            if subsample_size >= 1:
                subsample_size = int(subsample_size)
            else: # if subsample_size is a fraction
                subsample_size = int(round(subsample_size * len(dset.ids)))
            subsample = np.random.choice(dset.ids, subsample_size, replace=False)
            dset = dset.slice(subsample)
            dset.save(os.path.join(out_dir, 'dset.pkl.gz'))
    print(f"subsample {subsample_idx}" if (subsample_idx is not None) else "full sample")
    # run analysis
    if analysis == 'ALE':
        run_ale(dset, mask, out_dir, n_iters=n_iters, use_gpu=use_gpu, n_cores=n_cores)
    elif analysis == 'SALE':
        run_sale(dset, mask, out_dir, n_iters=n_iters, use_gpu=use_gpu, n_cores=n_cores)
    elif analysis == 'dSALE':
        run_sale(dset, mask, out_dir, n_iters=n_iters, use_gpu=use_gpu, n_cores=n_cores, deterministic=True)
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
    parser.add_argument('analysis', type=str, help='analysis to run', choices=['ALE', 'SALE', 'dSALE', 'k_cluster', 'macm'])
    parser.add_argument('bd', type=str, help='behavioural domain to run')
    parser.add_argument('subbd', type=str, help='sub-behavioural domain to run')
    parser.add_argument('-n_subsamples', type=int, default=0, help='number of subsamples to run'
                                                                   ' (0 indicates no subsampling)')        
    parser.add_argument('-subsample_size', type=float, default=None, help='number/fraction of experiments to subsample')
    parser.add_argument('-n_iters', type=int, default=10000, help='number of null permutations')
    parser.add_argument('-use_gpu', type=int, help='use gpu', default=True)
    parser.add_argument('-n_cores', type=int, help='number of CPU cores', default=4)
    # k_cluster arguments
    parser.add_argument('-mask', type=str, default='D2009_MNI', help='mask for k-clustering')
    parser.add_argument('-height', type=float, default=0.001, help='height threshold for k-clustering')
    parser.add_argument('-k', type=int, default=50, help='k size for k-clustering')
    # macm input path
    parser.add_argument('-macm_in', type=str, default=None, help='full path to thresholded z-map as seed of macm')

    args = parser.parse_args()

    # get default subsample size depending on bd vs subbd
    if (args.subsample_size is None) and (args.n_subsamples > 0):
        args.subsample_size = SUBSAMPLE_SIZE['BD' if args.subbd == args.bd else 'SubBD']

    # disable subsampling if subsample size is 0
    if args.subsample_size == 0:
        args.subsample_size = None

    # # check for GPU
    # if cuda.is_available():
    #     print("GPU found, using GPU")
    #     use_gpu = True
    # else:
    #     print("No GPU found, using CPU")
    #     use_gpu = False
    # the above does not work on CPU nodes
    use_gpu = args.use_gpu

    if args.analysis == 'k_cluster':
        sale_k_cluster(k=args.k, height=args.height, mask_name=args.mask)
    elif args.analysis == 'macm':
        run_macm(args.macm_in, n_iters=args.n_iters)
    else:
        run(analysis=args.analysis, bd=args.bd, subbd=args.subbd, 
            n_iters=args.n_iters, use_gpu=use_gpu, n_cores=args.n_cores, 
            subsample_size=args.subsample_size, n_subsamples=args.n_subsamples)
