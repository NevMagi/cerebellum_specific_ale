import nimare
import os
import nilearn.image
import nibabel
import argparse
import json

from run import run_sale
import utils

# define input and output directories
INPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'input')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')
# set minimum number of experiments to run meta-analysis
MIN_EXPERIMENTS = 15
# define brain GM mask
MASK_NAME = 'Grey10'
# define kernel sum map used for sampling null coordinates in pSALE
PROB_MAP_PATH = utils.get_kernels_sum(os.path.join(OUTPUT_DIR, 'BrainMap_dump_Feb2024.pkl.gz'))

def run_macm(zmap_path, n_iters=10000, use_gpu=True, n_cores=4, merge_exps=True):
    """
    Run MACM from (cerebellar) seed clusters in thresholded zmap to the whole brain
    and save the results to {zmap_prefix}_macm directory

    Parameters
    ----------
    zmap_path : str
        path to thresholded z-map
    n_iters : int
        number of null permutations
    use_gpu : bool
    n_cores : int
        number of CPU cores to use. Only used if use_gpu is False.
    merge_exps : bool
        whether to merge experiments within the same study
    """
    # define seed cluster(s)
    zmap = nibabel.load(zmap_path)
    seed = nilearn.image.binarize_img(zmap)
    # load mask
    mask_img = nibabel.load(os.path.join(INPUT_DIR, 'maps', f'{MASK_NAME}.nii.gz'))
    # load or create masked dump
    dump_path = os.path.join(OUTPUT_DIR, 'data', f'BrainMap_dump_Feb2024_mask-{MASK_NAME}.pkl.gz')
    # first time this function is called, load the whole-brain
    # dump and filter it to the cerebellar mask
    if not os.path.exists(dump_path):
        print("Masking the whole-brain dump to the GM mask...")
        dump_brain = nimare.dataset.Dataset.load(os.path.join(OUTPUT_DIR, 'data', 'BrainMap_dump_Feb2024.pkl.gz'))
        dump = utils.filter_coords_to_mask(dump_brain, mask_img)
        dump.save(dump_path)
    else:
        # load the masked dump if it already exists
        dump = nimare.dataset.Dataset.load(dump_path)
    # filter dump to experiments including a focus in the mask
    seed_dset = dump.slice(dump.get_studies_by_mask(seed))
    if merge_exps:
        # merge multiple experiments within the same study
        seed_dset = utils.merge_experiments(seed_dset)
    if len(seed_dset.ids) < MIN_EXPERIMENTS:
        print(f"Skipping {zmap_path} MACM due to insufficient number of experiments")
        return
    # set output path
    out_dir = zmap_path.replace('.nii.gz', '_macm')
    os.makedirs(out_dir, exist_ok=True)
    # save dset
    seed_dset.save(os.path.join(out_dir, 'dset.pkl.gz'))
    # save experiment stats
    with open(os.path.join(out_dir, 'stats.json'), 'w') as f:
        json.dump({
            'n_exp': seed_dset.ids.size,
            'n_foci': seed_dset.coordinates.shape[0],
            'n_subs': int(seed_dset.metadata['sample_sizes'].apply(lambda c: c[0]).sum())
        }, f, indent=4)
    # run pSALE
    run_sale(seed_dset, mask_img, out_dir,
             n_iters=n_iters, use_gpu=use_gpu, n_cores=n_cores,
             approach='probabilistic')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-zmap_path', type=str, required=True, help='full path to thresholded z-map as seed of macm')
    parser.add_argument('-n_iters', type=int, default=10000, help='number of null permutations')

    args = parser.parse_args()

    run_macm(args.zmap_path, n_iters=args.n_iters)