import nimare
import os
import numpy as np
import nilearn.image
import nilearn.maskers
import nibabel
from nimare.meta.cbma.ale import ALE, SCALE
from nimare_gpu.ale import DeviceALE, DeviceSCALE
from nimare.correct import FWECorrector
import argparse
import json

import utils

# define input and output directories
INPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'input')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')
# set minimum number of experiments to run meta-analysis / subsampling
MIN_EXPERIMENTS = 15
MIN_EXPERIMENTS_SUBSAMPLING = 60
# define dilated cerebellar mask (created via utils.create_Driedrischen2009_mask('MNI', dilation=6))
MASK_NAME = 'D2009_MNI_dilated-6mm'
# define proper (non-dilated) cerebellar mask (created via utils.create_Driedrischen2009_mask('MNI', dilation=0))
MASK_NAME_PROPER = 'D2009_MNI'

def prepare_data(bd_prefix, pure=False, merge_exps=True, out_path=None):
    """
    Creates the NiMARE dataset correponding to `bd_prefix` from the
    BrainMap dump. Note that this function assumes that the BrainMap
    data exists in the output directory as 'BrainMap_dump_Feb2024.pkl.gz'

    Parameters
    ----------
    bd_prefix : str
        The prefix of the behavioral domain to filter
        e.g. "Cognition" or "Cognition.Language"
    pure : bool
        Include experiments which are only associated with
        the given behavioral domain and no other domains
    merge_exps : bool
        Merge multiple experiments reported within the same study
    out_path : str
        Path to save the dataset to
    
    Returns
    -------
    dset : nimare.dataset.Dataset
    """
    print("Preparing data...")
    dump_path = os.path.join(OUTPUT_DIR, 'data', f'BrainMap_dump_Feb2024_mask-{MASK_NAME}.pkl.gz')
    # first time this function is called, load the whole-brain
    # dump and filter it to the cerebellar mask
    if not os.path.exists(dump_path):
        print("Masking the whole-brain dump to the cerebellar mask...")
        dump_brain = nimare.dataset.Dataset.load(os.path.join(OUTPUT_DIR, 'data', 'BrainMap_dump_Feb2024.pkl.gz'))
        mask_img = nibabel.load(os.path.join(INPUT_DIR, 'maps', f'{MASK_NAME}.nii.gz'))
        dump = utils.filter_coords_to_mask(dump_brain, mask_img)
        dump.save(dump_path)
    else:
        # load the masked dump if it already exists
        dump = nimare.dataset.Dataset.load(dump_path)
    # get the full list of BDs within the dump that
    # start with bd_prefix
    # e.g. 'Cognition' -> ['Cognition', 'Cognition.Language', ...]
    all_bds = sorted(list(set(dump.metadata['behavioral_domain'].sum())))
    selected_bds = set(filter(lambda s: s.startswith(bd_prefix), all_bds))
    # filter the dump to the experiments which include 
    # the selected behavioral domains (in the case
    # of `pure`, experiments which include *only* the selected
    # behavioral domains)
    if pure:
        exp_mask = dump.metadata['behavioral_domain'].map(lambda c: len(set(c).difference(selected_bds))==0)
    else:
        exp_mask = dump.metadata['behavioral_domain'].map(lambda c: len(set(c).intersection(selected_bds))>0)
    selected_ids = dump.metadata.loc[exp_mask, 'id'].values
    dset = dump.slice(selected_ids)
    # merge multiple contrasts (experiments) within the same study if indicated
    if merge_exps:
        dset = utils.merge_experiments(dset)
    # save the dataset if indicated
    if out_path is not None:
        dset.save(out_path)
    return dset

def prepare_data_neurosynth(term, frequency_threshold=0, out_path=None):
    """
    Creates the NiMARE dataset correponding to `bd_prefix` from the
    BrainMap dump. Note that this function assumes that the BrainMap
    data exists in the output directory as 'BrainMap_dump_Feb2024.pkl.gz'

    Parameters
    ----------
    term : str
        must be from https://github.com/neurosynth/neurosynth-data/blob/master/data-neurosynth_version-7_vocab-terms_vocabulary.txt
    frequency_threshold: float
        threshold of term frequency in the abstract for an experiment to be selected
    out_path : str
        Path to save the dataset to
    
    Returns
    -------
    dset : nimare.dataset.Dataset
    """
    print("Preparing data...")
    dump_path = os.path.join(OUTPUT_DIR, 'data', f'Neurosynth_dump_mask-{MASK_NAME}.pkl.gz')
    # first time this function is called, load the whole-brain
    # dump and filter it to the cerebellar mask
    if not os.path.exists(dump_path):
        print("Masking the whole-brain dump to the cerebellar mask...")
        dump_brain = nimare.dataset.Dataset.load(os.path.join(OUTPUT_DIR, 'data', 'Neurosynth_dump.pkl.gz'))
        mask_img = nibabel.load(os.path.join(INPUT_DIR, 'maps', f'{MASK_NAME}.nii.gz'))
        dump = utils.filter_coords_to_mask(dump_brain, mask_img)
        dump.save(dump_path)
    else:
        # load the masked dump if it already exists
        dump = nimare.dataset.Dataset.load(dump_path)
    # clean up terms table
    annots_clean = dump.annotations.copy()
    annots_clean.columns = list(annots_clean.columns[:3]) + list(annots_clean.columns[3:].str.replace('terms_abstract_tfidf__', ''))
    # select experiments with selected terms
    selected_ids = annots_clean.loc[annots_clean[term]>frequency_threshold, 'id'].values
    dset = dump.slice(selected_ids)
    # save the dataset if indicated
    if out_path is not None:
        dset.save(out_path)
    return dset

def run_ale(dset, mask, out_dir=None, n_iters=10000, use_gpu=True, n_cores=4):
    """
    Runs classic ALE on the given dataset and saves its results
    to the output directory

    Parameters
    ----------
    dset : nimare.dataset.Dataset
    mask : nilearn.maskers.NiftiMasker
    out_dir : str
    n_iters : int
        Number of iterations to use in cFWE correction
    use_gpu : bool
    n_cores : int
        Number of CPU cores to use. Only used if `use_gpu` is False 
    """
    print(f"Running ALE on ({len(dset.ids)} experiments)...")
    # create ALE or DeviceALE object depending on the hardware
    if use_gpu:
        meta = DeviceALE(mask=mask, null_method='approximate')
        n_cores = 1
    else:
        meta = ALE(mask=mask, null_method='approximate')
    # run true (observed) ALE
    results = meta.fit(dset)
    # run cFWE correction
    print("Running FWE correction...")
    corr = FWECorrector(
        method="montecarlo", 
        voxel_thresh=0.001, 
        n_iters=n_iters, 
        n_cores=n_cores, 
        vfwe_only=False
    )
    cres = corr.transform(results)
    # save results
    if out_dir is not None:
        print("Saving results...")
        results.save_maps(out_dir, 'uncorr')
        cres.save_maps(out_dir, 'cFWE')

def run_sale(dset, mask, out_dir=None, n_iters=10000, use_gpu=True, 
             n_cores=4, height_thr=0.001, k=50, sigma_scale=1.0, 
             debug=False, use_mmap=False, approach='probabilistic',
             source='BrainMap'):
    """
    Runs specific ALE (SALE) on the given dataset and saves its results.
    Note that probabilistic SALE is not implemented for CPU.

    Parameters
    ----------
    dset : nimare.dataset.Dataset
    mask : nilearn.maskers.NiftiMasker
    out_dir : str
    n_iters : int
        Number of iterations to use in SALE for p-value calculation
    use_gpu : bool
    n_cores : int
        Number of CPU cores to use. Only used if `use_gpu` is False
    height_thr : float
        Voxel-wise threshold for cluster-extent correction
    k : int
        Cluster size (n voxels) for cluster-extent correction
    sigma_scale : float
        Scaling factor for the Gaussian kernel used in SALE
        Only used if `use_gpu` is True
    debug : bool
        Keep the null permutations in memory for debugging purposes
        Only used if `use_gpu` is True
    use_mmap : bool
        Use memory mapping for the null permutations
        Will make it slower but use less memory
        Only used if `use_gpu` is True
    approach : {'probabilistic', 'deterministic'}
        Approach to use for sampling null coordinates
        - probabilistic: uses a null probability map
        - deterministic: uses a null set of coordinates
    source : {'BrainMap', 'Neurosynth'}
    """
    print(f"Running {approach} SALE on ({len(dset.ids)} experiments)...")
    if source == 'BrainMap':
        dump_prefix = 'BrainMap_dump_Feb2024'
    elif source == 'Neurosynth':
        dump_prefix = 'Neurosynth_dump'
    if approach == 'deterministic':
        # for dSALE the null set of coordinates are loaded
        prob_map = None
        xyz = utils.get_null_xyz(
            os.path.join(OUTPUT_DIR, 'data', f'{dump_prefix}_mask-{MASK_NAME}.pkl.gz'),
            os.path.join(INPUT_DIR, 'maps', f'{MASK_NAME}.nii.gz'), 
            unique=False
        )
    else:
        # for pSALE the null probability map of effects
        # (kernel sum) is loaded
        # it will be masked and normalized to sum of 1
        # in the DeviceSCALE class
        prob_map_path = utils.get_kernels_sum(os.path.join(OUTPUT_DIR, 'data', f'{dump_prefix}.pkl.gz'))
        prob_map = nibabel.load(prob_map_path)
        xyz = None
    if use_gpu:
        # initialize DeviceSCALE object
        meta = DeviceSCALE(
            approach,
            prob_map=prob_map,
            xyz=xyz,
            mask=mask, 
            n_iters=n_iters,
            keep_perm_nulls=debug,
            sigma_scale=sigma_scale,
            use_mmap=use_mmap,
        )
    else:
        if approach == 'deteministic':
            meta = SCALE(
                xyz=xyz,
                mask=mask,
                n_iters=n_iters,
                n_cores=n_cores,
            )
        else:
            raise NotImplementedError("CPU version of probabilistic SALE is not implemented")
    # run SALE and its permutations
    results = meta.fit(dset)
    if out_dir is not None:
        # save uncorrected results
        print("Saving results...")
        results.save_maps(out_dir, 'uncorr')
        # run cluster-extent correction after masking
        # the results to the cerebellar proper mask (not dilated)
        proper_mask_img = nibabel.load(os.path.join(INPUT_DIR, 'maps', f'{MASK_NAME_PROPER}.nii.gz'))
        # mask z and logp to the proper mask
        z_orig = results.get_map('z')
        logp_orig = results.get_map('logp')
        z_masked = nilearn.image.math_img("m * img", img=z_orig, m=proper_mask_img)
        logp_masked = nilearn.image.math_img("m * img", img=logp_orig, m=proper_mask_img)
        # cluster extent correction on masked results
        cres = utils.cluster_extent_correction({'z': z_masked, 'logp': logp_masked}, k=k, height_thr=height_thr)
        # save corrected results
        for map_name, img in cres.items():
            img.to_filename(os.path.join(out_dir, f'corr_cluster_h-{str(height_thr)[2:]}_k-{k}_mask-{MASK_NAME_PROPER}_{map_name}.nii.gz'))

def run_meta(analysis, source, subbd='', 
             n_iters=10000, use_gpu=True, n_cores=4, 
             subsample_size=0, subsample_idx=None, 
             n_subsamples=0):
    """
    Main function that runs main and subsampled ALE/SALE after
    preparing their datasets and saves the results to the output
    folder

    Parameters
    ----------
    analysis : {'ALE', 'SALE', 'dSALE'}
        - ALE: classic activation likelihood estimation
        - SALE: (probabilistic) specific activation likelihood estimation
        - dSALE: deterministic specific activation likelihood estimation
    source : {'BrainMap', 'Neurosynth"}
    subbd : str
        - Behavioral (sub)domain, e.g. 'Action' or 'Action.Execution' for BrainMap
        - Abstract term for Neurosynth
    n_iters : int
        Number of null permutations
    use_gpu : bool
    n_cores : int
        Number of CPU cores to use. Only used if `use_gpu` is False
    subsample_size : float
        Number/fraction of experiments to subsample
    subsample_idx : int
        Index of the subsample to run
        Is used to recursively call this function to run
        subsamples
    n_subsamples : int
        Number of subsamples to run. Only used if `subsample_size` is not None
    """
    if source == 'Neurosynth':
        # this is only going to be run once
        utils.prep_neurosynth()
    # replace space and slash with their respective strings
    # (a workaround for correctly passing subbds through condor and argparse
    # when they contain spaces and slashes)
    subbd = subbd.replace('_space_', ' ').replace('_slash_', '/')
    if source == 'BrainMap':
        bd = subbd.split('.')[0]
    else:
        bd = ''
    print(f"Running {analysis} ({source})", end=" ")
    print("on gpu" if use_gpu else f"on cpu ({n_cores} cores)")
    # prepare dset (load or create it)
    print(f"SubBD/Term: {subbd}")
    subbd_clean = subbd.replace(' ', '').replace('/', '') # used for subfolder names
    if source == 'Neurosynth':
        dset_path = os.path.join(OUTPUT_DIR, 'data', source, subbd_clean, 'dset.pkl.gz')
    else:
        dset_path = os.path.join(OUTPUT_DIR, 'data', bd, subbd_clean, 'dset.pkl.gz')
    if os.path.exists(dset_path):
        dset = nimare.dataset.Dataset.load(dset_path)
    else:
        os.makedirs(os.path.dirname(dset_path), exist_ok=True)
        if source == 'BrainMap':
            dset = prepare_data(subbd, out_path=dset_path)
        elif source == 'Neurosynth':
            dset = prepare_data_neurosynth(subbd, out_path=dset_path)
    # skip meta-analysis if there are not enough experiments
    if len(dset.ids) < MIN_EXPERIMENTS:
        print(f"Skipping {bd} {subbd} due to insufficient number of experiments")
        return
    if (subsample_size > 0) & (len(dset.ids) < MIN_EXPERIMENTS_SUBSAMPLING):
        print(f"Skipping {bd} {subbd} subsampling due to insufficient number of experiments")
        return
    # create output folder
    if source == 'Neurosynth':
        out_dir = os.path.join(OUTPUT_DIR, analysis, source, subbd_clean)
    else:
        out_dir = os.path.join(OUTPUT_DIR, analysis, bd, subbd_clean)
    if (subsample_size is not None) & (subsample_idx is not None):
        if subsample_size >= 1:
            # susample size is an integer showing N experiments
            out_dir = os.path.join(out_dir, f'subsamples_n-{int(subsample_size)}', str(subsample_idx))
        else:
            out_dir = os.path.join(out_dir, f'subsamples_p-{subsample_size}'.replace('.',''), str(subsample_idx))
    os.makedirs(out_dir, exist_ok=True)
    # load mask
    mask_img = nibabel.load(os.path.join(INPUT_DIR, 'maps', f'{MASK_NAME}.nii.gz'))
    mask = nilearn.maskers.NiftiMasker(mask_img)
    # subsample if indicated
    if subsample_size > 0:
        # run subsamples
        if subsample_idx is None:
            # in parent run of subsamples subsample_size is defined
            # but subsample index is not set, then this loops through
            # number of subsamples and recursively calls this function
            # for each subsample index, then exits (returns)
            for i in range(n_subsamples):
                run_meta(analysis=analysis, bd=bd, subbd=subbd, n_iters=n_iters, 
                    use_gpu=use_gpu, n_cores=n_cores, subsample_size=subsample_size, 
                    subsample_idx=i)
            return
        else:
            # i.e., this is not the parent run, and we should now run
            # meta-analysis on the subsample indicated by subsample_idx
            # slice the dataset to subsample and continue
            # use reproducable but variable seeds per subsample
            np.random.seed(1234+subsample_idx)
            if subsample_size >= 1:
                # susample size is an integer showing N experiments
                subsample_size = int(subsample_size)
            else:
                # subsample size is a fraction
                subsample_size = int(round(subsample_size * len(dset.ids)))
            # slice the dataset to subsample
            subsample = np.random.choice(dset.ids, subsample_size, replace=False)
            dset = dset.slice(subsample)
            dset.save(os.path.join(out_dir, 'dset.pkl.gz'))
    print(f"subsample {subsample_idx}" if (subsample_idx is not None) else "full sample")
    # fix the seed used for generating SALE/ALE nulls
    np.random.seed(0)
    # run analysis
    if analysis == 'ALE':
        run_ale(dset, mask, out_dir, n_iters=n_iters, use_gpu=use_gpu, n_cores=n_cores)
    elif analysis == 'SALE':
        run_sale(dset, mask, out_dir, n_iters=n_iters, use_gpu=use_gpu, n_cores=n_cores, source=source)
    elif analysis == 'dSALE':
        run_sale(dset, mask, out_dir, n_iters=n_iters, use_gpu=use_gpu, n_cores=n_cores, source=source, approach='deterministic')
    # save config after running the analysis
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
    # save experiment stats
    with open(os.path.join(out_dir, 'stats.json'), 'w') as f:
        json.dump({
            'n_exp': dset.ids.size,
            'n_foci': dset.coordinates.shape[0],
            'n_subs': int(dset.metadata['sample_sizes'].apply(lambda c: c[0]).sum())
        }, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('analysis', type=str, help='analysis to run', 
        choices=['ALE', 'SALE', 'dSALE', 'macm', 'variogram'])
    parser.add_argument('source', type=str, help='data source',
        choices=['BrainMap', 'Neurosynth']
    )
    parser.add_argument('subbd', type=str, 
                        help='BrainMap behavioural (sub)domain or Neurosynth term to run')
    parser.add_argument('-n_subsamples', type=int, default=0, 
                        help='number of subsamples to run (0 indicates no subsampling)')        
    parser.add_argument('-subsample_size', type=float, default=0, 
                        help='number/fraction of experiments to subsample')
    parser.add_argument('-n_iters', type=int, default=10000, 
                        help='number of null permutations')
    parser.add_argument('-use_gpu', type=int, help='use gpu', default=True)
    parser.add_argument('-n_cores', type=int, help='number of CPU cores', default=4)

    args = parser.parse_args()

    run_meta(analysis=args.analysis, source=args.source, subbd=args.subbd, 
        n_iters=args.n_iters, use_gpu=args.use_gpu, n_cores=args.n_cores, 
        subsample_size=args.subsample_size, n_subsamples=args.n_subsamples)
