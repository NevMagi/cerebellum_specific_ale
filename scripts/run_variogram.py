import os
import numpy as np
import nilearn.plotting
import nilearn.maskers
import nibabel
import argparse
from brainsmash.mapgen.sampled import Sampled

import utils

# define input and output directories
INPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'input')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')

def run_variogram(z_path, perms=1000):
    """
    Creates variogram-based surrogates of the uncorrected Z-maps 
    (meta-analysis outputs) and saves them as 
    {z_path}_2mm_mask-D2009_MNI_surrogates-variogram_n-{perms}.npy

    Parameters
    ----------
    z_path : str
        path to the unthresholded Z-map
    perms : int
        number of surrogates to generate
    """
    # determine paths
    surrogates_path = z_path.replace('.nii.gz', f'_2mm_mask-D2009_MNI_surrogates-variogram_n-{perms}.npy')
    if os.path.exists(surrogates_path):
        print(f"Surrogates already calculated for {z_path}")
        return
    # load or create sorted distance matrix of
    # all voxels within D2009 mask in 2mm MNI space
    npydfile = os.path.join(INPUT_DIR, 'maps', 'D2009_MNI_2mm_32bit_distmat_sorted.npy')
    npyifile = os.path.join(INPUT_DIR, 'maps', 'D2009_MNI_2mm_32bit_distmat_argsort.npy')
    if not (os.path.exists(npydfile) and os.path.exists(npyifile)):
        # create distance matrix which will create npydfile and npyifile
        utils.create_distmat(os.path.join(INPUT_DIR, 'maps', 'D2009_MNI.nii.gz'))
        assert os.path.exists(npydfile) and os.path.exists(npyifile) # make sure they are created
    # load 2mm mask and get number of in-mask voxels
    mask_2mm = nibabel.load(os.path.join(INPUT_DIR, 'maps', 'D2009_MNI_2mm.nii.gz'))
    nv = np.sum(np.isclose(mask_2mm.get_fdata(), 1))
    fpd = np.lib.format.open_memmap(
        npydfile, mode='r', dtype=np.float32, shape=(nv, nv)
    )
    fpi = np.lib.format.open_memmap(
        npyifile, mode='r', dtype=np.int32, shape=(nv, nv)
    )
    # load z-map and transform it to 2mm mask space
    z_nii = nibabel.load(z_path)
    z_2mm = nilearn.image.resample_to_img(z_nii, mask_2mm)
    z_2mm.to_filename(z_path.replace('.nii.gz', '_2mm.nii.gz'))
    # get within-mask values
    x = z_2mm.get_fdata()[np.isclose(mask_2mm.get_fdata(), 1)]
    # create surrogates
    gen = Sampled(x, fpd, fpi, resample=True, seed=0)
    surrogate_maps = gen(n=perms)
    # save them
    np.save(surrogates_path, surrogate_maps)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('zmap_path', type=str, help='full path to unthresholded z-map')
    parser.add_argument('-perms', type=int, default=1000, help='number of surrogates to generate')
    args = parser.parse_args()

    run_variogram(args.zmap_path, args.perms)