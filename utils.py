import os
import numpy as np
from tqdm import tqdm
import nimare
from nilearn._utils import load_niimg
import copy
from nimare.utils import mm2vox
import nibabel
import nilearn.image
from scipy import ndimage

INPUT_DIR = '/data/project/cerebellum_ale/input'
OUTPUT_DIR = '/data/project/cerebellum_ale/output'

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
    filtered_ids = filtered_dset.coordinates['id'].unique()
    filtered_dset = filtered_dset.slice(filtered_ids)
    return filtered_dset

def create_Driedrischen2009_mask(space='MNI', dilation=6):
    out_path = f'{INPUT_DIR}/maps/D2009_{space}'
    if dilation:
        out_path += f'_dilated-{dilation}mm'
    out_path += '.nii.gz'
    dseg = nibabel.load(f'{INPUT_DIR}/cerebellar_atlases/Diedrichsen_2009/atl-Anatom_space-{space}_dseg.nii')
    mask_img = nilearn.image.binarize_img(dseg)
    if dilation:
        dilated_mask_data = ndimage.binary_dilation(mask_img.get_fdata(), iterations=dilation).astype(float)
        mask_img = nilearn.image.new_img_like(mask_img, dilated_mask_data)
    mask_img.to_filename(out_path)

def get_null_xyz(mask_path=f'{INPUT_DIR}/maps/D2009_MNI.nii.gz', unique=False):
    xyz_path = mask_path.replace('.nii.gz', '_xyz.npy')
    if os.path.exists(xyz_path):
        # load previously created xyz
        xyz = np.load(mask_path.replace('.nii.gz', '_xyz.npy'))
    else:
        # load and mask source datsets
        mask_img = nibabel.load(mask_path)
        brain_dsets = {} # whole-brain
        dsets = {} # filtered to the cerebellum
        domains = ['Action', 'Cognition', 'Emotion', 'Perception', 'Interoception']
        for bd in domains:
            input_path = os.path.join(INPUT_DIR, 'sleuth', 'Behavioural_Domains', 'activations', bd, f'pos_bd-{bd.lower()}.txt')
            brain_dsets[bd] = nimare.io.convert_sleuth_to_dataset(input_path, target="mni152_2mm")
            dsets[bd] = filter_coords_to_mask(brain_dsets[bd], mask_img)
        # create xyz
        xyz = []
        for bd in domains:
            xyz.append(dsets[bd].coordinates[['x', 'y','z']].values)
        xyz = np.concatenate(xyz, axis=0)
        # save it
        np.save(xyz_path, xyz)
    if unique:
        xyz = np.unique(xyz, axis=0)
    return xyz

def cluster_extent_correction(res, height_thr=0.001, k=50):
    """
    Multiple comparisons correction using cluster-extent based thresholding
    (mainly for SALE).

    Parameters
    ----------
    res : :obj:`nimare.results.MetaResult`
        A MetaResult object containing voxel-wise p-values.
    height_thr : :obj:`float`
        Voxel-wise p-value threshold.
    k : :obj:`int`
        Cluster extent (number of voxels) threshold

    Returns
    -------
    maps : :obj:`dict` of :obj:`nibabel.Nifti1Image`
        including 'logp' and 'z' maps masked to significant clusters
    """
    # get -logp of threshold as voxel-wise p-values
    # are saved as -log10(p)
    logp_height_thr = -np.log10(height_thr)
    # get voxel-wise p-values image data
    logp_3d_img = res.get_map("logp")
    logp_3d = logp_3d_img.get_fdata()
    # identify clusters
    conn = ndimage.generate_binary_structure(rank=3, connectivity=1)
    labeled_arr3d, _ = ndimage.label(logp_3d > logp_height_thr, conn)
    # get cluster sizes
    clust_sizes = np.bincount(labeled_arr3d.flatten())
    # set the first cluster size (background 0s) to 0
    clust_sizes[0] = 0
    # identify clusters that are larger than k
    sig_clusters = np.where(clust_sizes > k)[0]
    # create a mask of significant clusters
    sig_clusters_mask = np.in1d(labeled_arr3d.flatten(), sig_clusters).reshape(logp_3d.shape)
    # mask logp and z maps
    logp_3d *= sig_clusters_mask
    z_3d = res.get_map("z").get_fdata()
    z_3d *= sig_clusters_mask
    # create nibabel images
    logp_3d_img = nilearn.image.new_img_like(logp_3d_img, logp_3d)
    z_3d_img = nilearn.image.new_img_like(logp_3d_img, z_3d)
    # return maps
    return {"logp": logp_3d_img, "z": z_3d_img}