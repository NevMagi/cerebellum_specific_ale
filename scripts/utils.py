import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import nimare
from nilearn._utils import load_niimg
import copy
from nimare.utils import mm2vox
import nibabel
import nibabel.processing
import nilearn.image
from scipy import ndimage
import gc

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
    dist = np.zeros(mask_ijk.shape[0], dtype=float) # reusing the same array to reduce memory usage
    for point_idx in tqdm(range(dset_ijk.shape[0])): 
        # using a for loop instead of scipy cdist
        # to minimize memory usage
        dist[:] = np.linalg.norm(np.array(dset_ijk[point_idx])[np.newaxis, :] - mask_ijk, axis=1)
        in_mask[point_idx] = np.any(dist == 0)
    filtered_dset = copy.deepcopy(dset)
    filtered_dset.coordinates = dset.coordinates.loc[in_mask]
    filtered_ids = filtered_dset.coordinates['id'].unique()
    filtered_dset = filtered_dset.slice(filtered_ids)
    return filtered_dset

def merge_experiments(dset):
    """
    Merges multiple experiments (contrasts) of the same study
    into a single experiment.
    """
    # merge metadata
    metadata_merged = pd.DataFrame(columns=dset.metadata.columns[:6])
    for i, (study_id, study_metadata) in enumerate(dset.metadata.groupby('study_id')):
        metadata_merged.loc[i, :] = {
            'id': f'{study_id}-0',
            'study_id': study_id,
            'contrast_id': 0,
            'sample_sizes': [int(study_metadata['sample_sizes'].apply(lambda c: c[0]).mean().round())],
            'author': study_metadata.iloc[0].loc['author'],
            'year': study_metadata.iloc[0].loc['year']
        }
    # merge coordinates
    coordinates_merged = []
    for i, (study_id, study_coordinates) in enumerate(dset.coordinates.groupby('study_id')):
        curr_coordinates = study_coordinates.copy()
        curr_coordinates.loc[:, 'id'] = f'{study_id}-0'
        curr_coordinates.loc[:, 'contrast_id'] = 0
        coordinates_merged.append(curr_coordinates)
    coordinates_merged = pd.concat(coordinates_merged)
    # create annotations, images and texts dataframes
    # (probably not needed)
    annotations_merged = metadata_merged.iloc[:, :3]
    # merge IDs
    ids_merged = metadata_merged.loc[:, 'id'].unique()
    # apply the merging on the dataset and update it
    dset.metadata = metadata_merged
    dset.coordinates = coordinates_merged
    dset.annotations = annotations_merged
    ## .ids cannot be set so here we use slice
    dset = dset.slice(ids_merged)
    return dset

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

def get_null_xyz(null_dset_path, mask_path, unique=False):
    xyz_path = mask_path.replace('.nii.gz', '_xyz')
    if unique:
        xyz_path += '-unique'
    xyz_path += '.npy'
    if os.path.exists(xyz_path):
        # load previously created xyz
        xyz = np.load(xyz_path)
    else:
        # load and mask source datsets
        mask_img = nibabel.load(mask_path)
        dset = nimare.dataset.Dataset.load(null_dset_path)
        dset = filter_coords_to_mask(dset, mask_img)
        # create xyz
        xyz = dset.coordinates[['x', 'y', 'z']].values
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
    res : :obj:`nimare.results.MetaResult` or dict of ('logp', 'z') images
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
    if isinstance(res, dict):
        logp_3d_img = res["logp"]
        z_3d_img = res["z"]
    else:
        logp_3d_img = res.get_map("logp")
        z_3d_img = res.get_map("z")
    # get -logp of threshold as voxel-wise p-values
    # are saved as -log10(p)
    logp_height_thr = -np.log10(height_thr)
    # get voxel-wise p-values image data
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
    z_3d = z_3d_img.get_fdata()
    z_3d *= sig_clusters_mask
    # create nibabel images
    logp_3d_img = nilearn.image.new_img_like(logp_3d_img, logp_3d)
    z_3d_img = nilearn.image.new_img_like(logp_3d_img, z_3d)
    # return maps
    return {"logp": logp_3d_img, "z": z_3d_img}

def create_cerebellum_distmat():
    mask_path = '/data/project/cerebellum_ale/input/maps/D2009_MNI.nii.gz'
    mask_prefix = mask_path.replace('.nii.gz', '').replace('.nii', '')
    mask = nibabel.load(mask_path)
    # resample mask to 2mm
    mask_2mm = nibabel.processing.resample_to_output(mask, 2.0)
    mask_2mm.to_filename(mask_prefix+'_2mm.nii.gz')
    # get ijk of within-mask voxels
    mask_ijk = np.array(np.where(np.isclose(mask_2mm.get_fdata(), 1))).T
    # create distmat and store it as memmap
    distmat_path = mask_prefix + '_2mm_distmat.npy'
    n_points = mask_ijk.shape[0]
    distmat = np.lib.format.open_memmap(
        distmat_path, 
        mode='w+', 
        dtype=np.float64, 
        shape=(n_points, n_points)
    )
    scipy.spatial.distance.cdist(mask_ijk, mask_ijk, 'euclidean', out=distmat)
    # convert it to float32
    distmat32 = distmat.astype(np.float32)
    distmat32.tofile(mask_prefix + '_2mm_32bit_distmat.npy')
    # transform to sorted format expected by BrainSmash
    npydfile = mask_prefix + '_2mm_32bit_distmat_sorted.npy'
    npyifile = mask_prefix + '_2mm_32bit_distmat_argsort.npy'
    fpd = np.lib.format.open_memmap(
        npydfile, mode='w+', dtype=np.float32, shape=(n_points, n_points))
    fpi = np.lib.format.open_memmap(
        npyifile, mode='w+', dtype=np.int32, shape=(n_points, n_points))
    for row in tqdm(range(nv)):
        d = distmat32[row]
        sort_idx = np.argsort(d)
        fpd[row, :] = d[sort_idx]
        fpi[row, :] = sort_idx        