import numpy as np
from tqdm import tqdm
from nilearn._utils import load_niimg
import copy
from nimare.utils import mm2vox

def filter_coords_to_mask(dset, mask):
    """Filters the coordinates to the mask.

    Parameters
    ----------
    mask : img_like
        Mask across which to search for coordinates.

    Returns
    -------
    filtered_dset : :obj:`nimare.dataset.Dataset`
        A Dataset object only including the coordinates within the mask
    """
    # partially based on nimare.datasets.get_studies_by_mask
    mask = load_niimg(mask)

    dset_mask = dset.masker.mask_img
    if not np.array_equal(dset_mask.affine, mask.affine):
        print("Mask affine does not match Dataset affine. Assuming same space.")

    dset_ijk = mm2vox(dset.coordinates[["x", "y", "z"]].values, mask.affine)
    mask_ijk = np.vstack(np.where(mask.get_fdata())).T
    in_mask = np.zeros(dset_ijk.shape[0]).astype(bool)
    for point_idx in tqdm(range(dset_ijk.shape[0])): 
        # using a for loop instead of scipy cdist
        # to minimize memory usage
        dist = np.linalg.norm(np.array(dset_ijk[point_idx])[np.newaxis, :] - mask_ijk, axis=1)
        in_mask[point_idx] = np.any(dist == 0)
    filtered_dset = copy.deepcopy(dset)
    filtered_dset.coordinates = dset.coordinates.loc[in_mask]
    return filtered_dset