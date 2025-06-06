import os
import tempfile
import numpy as np
import pandas as pd
from tqdm import tqdm
import nimare
import copy
from nimare.utils import mm2vox
from nimare.extract import fetch_neurosynth
from nimare.io import convert_neurosynth_to_dataset
import nibabel
import nibabel.processing
import nilearn.image
import scipy
import SUITPy.flatmap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
import brainspace.mesh
import brainspace.plotting
import pyvirtualdisplay


INPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'input')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')
TOOLS_DIR = os.path.join(os.path.dirname(__file__), '..', 'tools')

##########################
# BrainMap domain labels #
##########################
bd_labels = pd.Series({
    'Action': 'Action',
    'Cognition': 'Cognition',
    'Perception': 'Perception',
    'Emotion': 'Emotion',
    'Interoception': 'Interoception',
    'Action.MotorLearning': 'Motor Learning',
    'Action.Observation': 'Observation',
    'Action.Execution.Speech': 'Speech Execution',
    'Action.Inhibition': 'Inhibition',
    'Action.Execution': 'Execution',
    'Action.Preparation': 'Preparation',
    'Action.Imagination': 'Imagination',
    'Cognition.Spatial': 'Spatial',
    'Cognition.SocialCognition': 'Social Cognition',
    'Cognition.Attention': 'Attention',
    'Cognition.Language.Syntax': 'Syntax',
    'Cognition.Memory': 'Memory',
    'Cognition.Memory.Explicit': 'Explicit Memory',
    'Cognition.Language': 'Language',
    'Cognition.Language.Semantics': 'Semantics',
    'Cognition.Memory.Working': 'Working Memory',
    'Cognition.Language.Phonology': 'Phonology',
    'Cognition.Reasoning': 'Reasoning',
    'Cognition.Temporal': 'Temporal',
    'Cognition.Language.Speech': 'Speech',
    'Cognition.Language.Orthography': 'Orthography',
    'Cognition.Music': 'Music',
    'Emotion.Positive.RewardGain': 'Reward/ Gain',
    'Emotion.Negative.Disgust': 'Disgust',
    'Emotion.Positive': 'Positive Emotion',
    'Emotion.Negative.Fear': 'Fear',
    'Emotion.Positive.Happiness': 'Happiness',
    'Emotion.Negative.Sadness': 'Sadness',
    'Emotion.Negative': 'Negative Emotion',
    'Emotion.Valence': 'Valence',
    'Emotion.Negative.Anger': 'Anger',
    'Emotion.Negative.Anxiety': 'Anxiety',
    'Perception.Vision': 'Vision',
    'Perception.Olfaction': 'Olfaction',
    'Perception.Vision.Motion': 'Vision - Motion',
    'Perception.Gustation': 'Gustation',
    'Perception.Somesthesis.Pain': 'Pain',
    'Perception.Audition': 'Audition',
    'Perception.Somesthesis': 'Somesthesis',
    'Perception.Vision.Color': 'Vision - Color',
    'Perception.Vision.Shape': 'Vision - Shape',
    'Interoception.Sexuality': 'Sexuality',
    'Interoception.RespirationRegulation': 'Respiration Regulation',
    'Interoception.Hunger': 'Hunger'
})
domains_list = ['Action', 'Cognition', 'Emotion', 'Interoception', 'Perception']
subdomains_with_domains = sorted(list(set(bd_labels.index) - set(domains_list)))
subdomains_list = ['.'.join(s.split('.')[1:]) for s in subdomains_with_domains]

neurosynth_terms = ['action', 'adaptation', 'anticipation', 'anxiety', 'arousal', 'attention', 'autobiographical memory', 'balance', 'belief', 'cognitive control', 'communication', 'competition', 'concept', 'consciousness', 'consolidation', 'context', 'coordination', 'decision', 'decision making', 'detection', 'eating', 'emotion', 'emotion regulation', 'empathy', 'encoding', 'episodic memory', 'expectancy', 'extinction', 'face recognition', 'facial expression', 'fear', 'fixation', 'focus', 'gaze', 'goal', 'imagery', 'induction', 'inference', 'inhibition', 'integration', 'intelligence', 'intention', 'interference', 'knowledge', 'language', 'language comprehension', 'learning', 'listening', 'localization', 'loss', 'maintenance', 'manipulation', 'meaning', 'memory', 'memory retrieval', 'mental imagery', 'morphology', 'motor control', 'movement', 'naming', 'navigation', 'object recognition', 'pain', 'perception', 'planning', 'priming', 'reading', 'reasoning', 'recall', 'recognition', 'reinforcement learning', 'response inhibition', 'response selection', 'retention', 'retrieval', 'reward anticipation', 'rhythm', 'risk', 'rule', 'salience', 'search', 'selective attention', 'semantic memory', 'sentence comprehension', 'skill', 'sleep', 'social cognition', 'spatial attention', 'speech perception', 'speech production', 'strategy', 'strength', 'sustained attention', 'task difficulty', 'thought', 'valence', 'verbal fluency', 'visual attention', 'visual perception', 'word recognition', 'working memory']
##############################################
# Helper functions for running meta-analyses #
##############################################
def filter_coords_to_mask(dset, mask):
    """
    Filters the coordinates to the mask.

    Parameters
    ----------
    dset : nimare.dataset.Dataset
    mask : img_like

    Returns
    -------
    filtered_dset : nimare.dataset.Dataset
        A Dataset object only including the coordinates within the mask
    """
    # partially based on nimare.datasets.get_studies_by_mask
    # Only load mask if it is a file path
    if isinstance(mask, str):
        mask = nibabel.load(mask)
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
    into a single experiment (per study)

    Parameters
    ----------
    dset : nimare.dataset.Dataset
        The Dataset object to merge.

    Returns
    -------
    dset : nimare.dataset.Dataset
        The Dataset object with merged experiments.
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

def create_Diedrichsen2009_mask(space='MNI', dilation=6):
    """
    Create a mask of the cerebellum based on the Diedrichsen 2009 atlas.

    Parameters
    ----------
    space : {'MNI', 'MNISym', 'SUIT'}
    dilation : int, optional
        if provided, the mask will be dilated by this amount (in mm)
    
    Returns
    -------
    out_path : str
        Path to the created mask
    """
    # specify output file name
    out_path = os.path.join(INPUT_DIR, 'maps', f'D2009_{space}')
    if dilation:
        out_path += f'_dilated-{dilation}mm'
    out_path += '.nii.gz'
    # load the atlas
    dseg = nibabel.load(f'{INPUT_DIR}/cerebellar_atlases/Diedrichsen_2009/atl-Anatom_space-{space}_dseg.nii')
    # binarize it to get a mask of cerebellum
    mask_img = nilearn.image.binarize_img(dseg)
    # dilate it if indicated
    if dilation:
        dilated_mask_data = scipy.ndimage.binary_dilation(mask_img.get_fdata(), iterations=dilation).astype(float)
        mask_img = nilearn.image.new_img_like(mask_img, dilated_mask_data)
    # save the mask
    mask_img.to_filename(out_path)
    return out_path

def get_kernels_sum(dset_path):
    """
    Calculate kernels sum map from the given dataset (e.g. BrainMap dump)
    which will be used to calculate null sampling probability in probabilistic SALE.
    
    It invovles two steps:
        1. Convolve each focus with its kernel, with a FWHM related to its sample size
        2. Add up all the kernels at the correct location, based on the ALE code, but using "sum" instead of "max"
    
    Parameters
    ----------
    dset_path : str
        Path to the dataset

    Returns
    -------
    out_path : str
        Path to the created kernels sum
    """
    out_path = dset_path.replace('.pkl.gz', '_kernels_sum.nii.gz')
    # load it if it's already created
    if os.path.exists(out_path):
        return out_path
    # otherwise create it
    # load the dataset
    dset = nimare.dataset.Dataset.load(dset_path)
    # precalculate all kernels for the sample sizes
    # that exist in dump experiments
    kernels = {}
    sample_sizes = sorted(dset.metadata['sample_sizes'].apply(lambda c: c[0]).unique())
    for sample_size in sample_sizes:
        _, kernels[sample_size] = nimare.meta.utils.get_ale_kernel(dset.masker.mask_img, sample_size)
    # create a ALE meta object for this dataset 
    # only for its helper functions (e.g. converting XYZ to IJK)
    meta = nimare.meta.ale.ALE()
    meta._collect_inputs(dset)
    meta._preprocess_input(dset)
    # calculate a sum of all the kernels
    # this is based on nimare's code for
    # calculating the MA maps but is modified
    # to take a sum of them rather than a max
    kernels_sum = np.zeros(dset.masker.mask_img.shape)
    for i, id in tqdm(enumerate(dset.ids)):
        # get experiment ijk and kernel
        ijk = meta.inputs_['coordinates'].loc[meta.inputs_['coordinates']['id']==id, ['i', 'j', 'k']].values
        sample_size = dset.metadata.iloc[i]['sample_sizes'][0]
        kernel = kernels[sample_size]
        # loop through foci and add up their kernels
        # centered at ijk of each focus
        mid = int(np.floor(kernel.shape[0] / 2.0))
        mid1 = mid + 1
        for j_peak in range(ijk.shape[0]):
            i, j, k = ijk[j_peak, :]
            xl = max(i - mid, 0)
            xh = min(i + mid1, kernels_sum.shape[0])
            yl = max(j - mid, 0)
            yh = min(j + mid1, kernels_sum.shape[1])
            zl = max(k - mid, 0)
            zh = min(k + mid1, kernels_sum.shape[2])
            xlk = mid - (i - xl)
            xhk = mid - (i - xh)
            ylk = mid - (j - yl)
            yhk = mid - (j - yh)
            zlk = mid - (k - zl)
            zhk = mid - (k - zh)
            if (
                (xl >= 0)
                & (xh >= 0)
                & (yl >= 0)
                & (yh >= 0)
                & (zl >= 0)
                & (zh >= 0)
                & (xlk >= 0)
                & (xhk >= 0)
                & (ylk >= 0)
                & (yhk >= 0)
                & (zlk >= 0)
                & (zhk >= 0)
            ):
                kernels_sum[xl:xh, yl:yh, zl:zh] += kernel[xlk:xhk, ylk:yhk, zlk:zhk]
    # convert to an image
    kernels_sum_img = nilearn.image.new_img_like(dset.masker.mask_img, kernels_sum)
    # save the image
    kernels_sum_img.to_filename(out_path)
    return out_path

def get_null_xyz(null_dset_path, mask_path, unique=False):
    """
    Get the coordinates of all the experiments within the null dataset
    (used for deterministic SALE)

    Parameters
    ----------
    null_dset_path : str
        Path to the null dataset
    mask_path : str
        Path to the mask
    unique : bool, optional
        If True, only unique coordinates are returned
    
    Returns
    -------
    xyz : np.ndarray
        The coordinates of all the experiments within the null dataset
    """
    # determine output path
    xyz_path = mask_path.replace('.nii.gz', '_xyz')
    if unique:
        xyz_path += '-unique'
    xyz_path += '.npy'
    # load previously created xyz if it exists
    if os.path.exists(xyz_path):
        
        xyz = np.load(xyz_path)
    # otherwise create it
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
    res : nimare.results.MetaResult or dict of ('logp', 'z') images
    height_thr : float
        Voxel-wise p-value threshold.
    k : int
        Cluster extent (number of voxels) threshold

    Returns
    -------
    maps : dict of nibabel.Nifti1Image
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
    conn = scipy.ndimage.generate_binary_structure(rank=3, connectivity=1)
    labeled_arr3d, _ = scipy.ndimage.label(logp_3d > logp_height_thr, conn)
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

def prep_neurosynth():
    """
    Downloads neurosynth data, converts it to a NiMARE dataset,
    calculates its kernel sum, masks it to the cerebellum,
    and saves all to <OUTPUT_DIR>/data
    """
    if os.path.exists(
        os.path.join(OUTPUT_DIR, 'data', 'Neurosynth_dump_mask-D2009_MNI_dilated-6mm.pkl.gz')
    ):
        print("Neurosynth data exists")
        return
    # fetch raw neurosynth data
    tmp_out_dir = tempfile.mkdtemp()
    neurosynth_db = fetch_neurosynth(
        data_dir=tmp_out_dir,
        version="7",
        overwrite=False,
        source="abstract",
        vocab="terms",
    )[0]
    # convert it to a NiMARE dataset
    neurosynth_dset = convert_neurosynth_to_dataset(
        coordinates_file=neurosynth_db["coordinates"],
        metadata_file=neurosynth_db["metadata"],
        annotations_files=neurosynth_db["features"],
    )
    # Neurosynth dataset does not include the sample size, which is needed throughout our 
    # meta-analysis pipeline. Therefore we'll assume that all experiments have a sample 
    # size of 20 which is a reasonable estimate according to the medians reported 
    # in Poldrack et al. 2017 (https://doi.org/10.1038/nrn.2016.167)
    neurosynth_dset.metadata['sample_sizes'] = 20
    neurosynth_dset.metadata['sample_sizes'] = neurosynth_dset.metadata['sample_sizes'].apply(lambda c: [c])
    # save the unfiltered version of neurosynth dataset
    neurosynth_dset.save(os.path.join(OUTPUT_DIR, 'data', 'Neurosynth_dump.pkl.gz'))
    # calculate kernels sum
    kernels_sum_path = get_kernels_sum(os.path.join(OUTPUT_DIR, 'data', 'Neurosynth_dump.pkl.gz'))
    # filter to the cerebellar mask and save the filtered dataset
    mask = nibabel.load(os.path.join(INPUT_DIR, 'maps', 'D2009_MNI_dilated-6mm.nii.gz'))
    neurosynth_dset_masked = filter_coords_to_mask(neurosynth_dset, mask)
    neurosynth_dset_masked.save(
        os.path.join(OUTPUT_DIR, 'data', 'Neurosynth_dump_mask-D2009_MNI_dilated-6mm.pkl.gz')
    )
    

    

##########################################
# Helper functions for statistical tests #
##########################################
def create_distmat(mask_path):
    """
    Creates sorted distance matrix of all voxels within the provided
    mask after resampling it to 2mm. The distance matrix is saved
    as a memmap file.

    Parameters
    ----------
    mask_path : str
        Path to the (cerebellar) mask.
    """
    print(f"Creating distance matrix of {mask_path}...")
    mask_prefix = mask_path.replace('.nii.gz', '').replace('.nii', '')
    mask = nibabel.load(mask_path)
    # resample mask to 2mm
    mask_2mm = nibabel.processing.resample_to_output(mask, 2.0)
    mask_2mm.to_filename(mask_prefix+'_2mm.nii.gz')
    # get ijk of within-mask voxels
    mask_ijk = np.array(np.where(np.isclose(mask_2mm.get_fdata(), 1))).T
    # create distmat and store it as memmap
    # (avoids using memory, as this matrix ends up being huge)
    distmat_path = mask_prefix + '_2mm_distmat.npy'
    n_points = mask_ijk.shape[0]
    distmat = np.lib.format.open_memmap(
        distmat_path, 
        mode='w+', 
        dtype=np.float64, 
        shape=(n_points, n_points)
    )
    # calculate eucliean distance between all points
    # Note: using xyz vs ijk does not make a difference
    # here since distance is relative
    scipy.spatial.distance.cdist(mask_ijk, mask_ijk, 'euclidean', out=distmat)
    # convert it to float32
    distmat32 = distmat.astype(np.float32)
    distmat32.tofile(mask_prefix + '_2mm_32bit_distmat.npy')
    # transform to sorted format expected by BrainSmash
    # this is doen similar to
    # https://github.com/murraylab/brainsmash/blob/f6a9c375ba2e591acbc2edc161fbcff12609749d/brainsmash/mapgen/memmap.py#L14
    # to match the format expected by BrainSmash
    npydfile = mask_prefix + '_2mm_32bit_distmat_sorted.npy'
    npyifile = mask_prefix + '_2mm_32bit_distmat_argsort.npy'
    fpd = np.lib.format.open_memmap(
        npydfile, mode='w+', dtype=np.float32, shape=(n_points, n_points))
    fpi = np.lib.format.open_memmap(
        npyifile, mode='w+', dtype=np.int32, shape=(n_points, n_points))
    for row in tqdm(range(n_points)):
        d = distmat32[row]
        sort_idx = np.argsort(d)
        fpd[row, :] = d[sort_idx]
        fpi[row, :] = sort_idx       

def create_linkage_matrix(model):
    """
    Creates linkage matrix from clustering `model`
    """
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, (left_child, right_child) in enumerate(model.children_):
        count_left = 1 if left_child < n_samples else counts[left_child - n_samples]
        count_right = 1 if right_child < n_samples else counts[right_child - n_samples]
        counts[i] = count_left + count_right
    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
    return linkage_matrix

def hierarchical_clustering(data, no_plot=True, ax=None):
    """
    Performs hierarchical clustering on the provided data and returns
    the clustering plus reordered indices, and plots dendrogram if indicated

    Parameters
    ----------
    data: (pd.DataFrame)
        Data with Shape (n_samples, n_features)
    no_plot: (bool)
    ax: matplotlib.axes.Axes

    Returns
    ------
    model, dendrogram, leaf_order
    """
    # hierarchical clustering
    model = AgglomerativeClustering(n_clusters=None, distance_threshold=0)
    model.fit(data)
    # linkage matrix
    linkage_matrix = create_linkage_matrix(model)
    if (not no_plot) and (ax is None):
        fig, ax = plt.subplots(figsize=(15, 2))
    dendro = dendrogram(linkage_matrix, show_contracted=True, ax=ax, no_labels=True, no_plot=no_plot)
    # extract the ordering
    leaf_order = dendro['leaves']
    return model, dendro, leaf_order

######################
# Plotting functions #
######################
def get_flatmap_mask_boundary_coords(mask):
    """
    Returns flatmap mesh coordinates corresponding to the boundaries
    of a `mask` (binary map) which must be a cerebellar flatmap.
    The boundary points can optionally be interpolated into more
    points to make the boundaries more visible. This is just done
    for visualization purposes and to make the boundary clearer.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask of the cerebellar flatmap in SUIT space

    Returns
    -------
    boundary_vert_coords : np.ndarray
        Coordinates of the boundary points
    """
    # load the flatmap mesh
    mesh = nilearn.surface.load_surf_mesh(os.path.join(TOOLS_DIR, 'SUITPy', 'SUITPy', 'surfaces', 'FLAT.surf.gii'))
    # find boundary faces (faces in which one and only one of
    # the triangle vertices are in the mask)
    boundary_faces = (mask[mesh.faces].sum(axis=1) == 1)
    # get all vertices that are at the boundary
    # AND are in the mask
    mask_boundary_verts = list(
        set(mesh.faces[boundary_faces].flatten()) & \
        set(np.where(mask)[0])
    )
    boundary_vert_coords = mesh.coordinates[mask_boundary_verts]
    # interpolate new points right in the middle of neighboring
    # vertices (in boundary) for two rounds
    for i in range(2):
        # first identify the neighbors of each boundary point (that are also on the boundary)
        # with a distance less than 5 mm
        distmat = scipy.spatial.distance_matrix(boundary_vert_coords, boundary_vert_coords)
        distmat[distmat == 0] = np.NaN
        origs, neighbors = np.where(distmat < 5)
        # add new points right in the middle of each boundary point and its neighbors (on the boundary)
        # the coordinates of these points should be the average of original vertex coordinates
        # and all its neighbors coordinates
        new_points = (boundary_vert_coords[origs] + boundary_vert_coords[neighbors]) / 2
        boundary_vert_coords = np.concatenate([boundary_vert_coords, new_points], axis=0)
    return boundary_vert_coords

def plot_and_save_histogram(values, title, color, save_path):
    """
    Plot and save a histogram of the provided values

    Parameters
    ----------
    values : np.ndarray
        Values to plot
    title : str
        Title of the plot
    color : str
        Color of the plot
    save_path : str
        Path to save the plot
    """
    plt.figure(figsize=(6, 4))
    sns.histplot(values, bins=20, kde=True, color=color)
    plt.title(title, fontsize=18, x = 0.5, y=1.02)
    plt.xlabel('Correlation Coefficient', fontsize = 14)
    plt.ylabel('Frequency', fontsize = 14)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_subsample_correlations(
    subsampling_settings, 
    corrs_long, 
    order_list, 
    stats, 
    save_dir=None, 
    data_type=None, 
    domain_colors_dict=None, 
    subdomain_labels=None, 
    fig_number=None
):
    """
    Plots spatial correlations for specified subsampling settings and saves the figures.

    Parameters
    ----------
    subsampling_settings : list
        List of subsampling settings to plot, e.g., ['n-25', 'n-50', 'p-02'].
    corrs_long : dict
        Dictionary containing DataFrames with correlation data. Expected keys are 'domains' and 'subdomains'.
    order_list : list
        List of domain or subdomain names in the order they should appear on the plot's x-axis.
    stats : DataFrame
        DataFrame containing additional statistics for each domain or subdomain, with column 'n_exp'.
    save_dir : str
        Directory path where the output figures should be saved.
    data_type : str, optional (default='domain')
        Specifies the type of data being plotted, either 'domain' or 'subdomain'.
    domain_colors_dict : dict
        Dictionary mapping domains to color codes for plotting.
    subdomain_labels : Series or dict
        Series or dictionary mapping original subdomain names to their desired display labels.
    fig_number : list of str, optional
        List of prefixes for filenames, e.g., ['4e_', '4f_'].

    Returns
    -------
    None
    """
    # Select the appropriate DataFrame from corrs_long based on data_type
    if data_type == 'subdomain':
        corr_data = corrs_long.get('subdomains')
        col_name = 'domain'
    else:
        corr_data = corrs_long.get('domains')
        col_name = 'domain'
    # Check if the selected DataFrame is available and contains the expected column
    if corr_data is None or col_name not in corr_data.columns:
        print(f"The specified data_type '{data_type}' does not have a corresponding DataFrame or column in corrs_long. Aborting function.")
        return
    # Loop over each subsample setting
    for idx, subsample in enumerate(subsampling_settings):
        # Create a new figure
        fig, ax = plt.subplots(figsize=(9, 6))
        # Make the background transparent
        fig.patch.set_facecolor('none')
        ax.patch.set_facecolor('none')
        # Filter data for the current subsampling type
        data_filtered = corr_data[corr_data['subsampling'] == subsample]
        # Skip plotting if there are no data points for this subsample
        if data_filtered.empty:
            print(f"No data available for subsample '{subsample}'. Skipping...")
            continue
        # Use colors based on the correct column type (domain or subdomain)
        colors = [domain_colors_dict.get(item.split('.')[0], '#333333') for item in data_filtered[col_name].unique()]
        # Plot stripplot and boxplot for the current subsampling type
        sns.stripplot(data=data_filtered, x=col_name, y='corr',
                      order=order_list, jitter=0.2,
                      s=1, alpha=0.8, ax=ax, hue=col_name, palette=colors, legend=False)
        sns.boxplot(data=data_filtered, x=col_name, y='corr',
                    order=order_list,
                    showfliers=False, showcaps=True, width=0.4,
                    boxprops={"facecolor": (1, 1, 1, 1)}, ax=ax, hue=col_name, palette=colors, legend=False)
        # Set plot labels and title
        ax.set_xlabel('')
        ax.set_ylabel('Correlation of 50 subsamples')
        # Determine the label format based on subsample setting
        if subsample.startswith('n-'):
            subsample_value = int(subsample.split("-")[1])  # For 'n-25' becomes '25'
            ax.set_title(f'N$_{{subsample}}$ = {subsample_value}', y=1.1, fontsize=25)
        elif subsample.startswith('p-'):
            subsample_value = float(subsample.split("-")[1]) / 10  # For 'p-02' becomes '0.2'
            ax.set_title(f'N$_{{subsample}}$ = {subsample_value} * N$_{{{col_name}}}$', y=1.1, fontsize=25)
        # Create x-axis labels with sample size information
        x_ticks_labels = []
        for item in order_list:
            if data_type == 'subdomain':
                # Get display label from subdomain_labels
                display_label = subdomain_labels.loc[item] if item in subdomain_labels.index else item
            else:
                display_label = item
            if subsample.startswith('p-'):
                # Calculate the sample size as the product of the percentage and the N for the domain or subdomain
                percentage = float(subsample_value)  # Keep it as 0.2, 0.4, etc.
                calculated_n = round(percentage * stats.loc[item, 'n_exp'])
                total_n = stats.loc[item, 'n_exp']  # Overall N for the domain or subdomain
                if data_type == 'subdomain':
                    # Format: "Display Label | calculated_n / total_n"
                    x_ticks_labels.append(f'{display_label} | {calculated_n} / {total_n}')
                else:
                    # Format: "Domain\ncalculated_n / total_n"
                    x_ticks_labels.append(f'{item}\n{calculated_n} / {total_n}')
            else:
                # For 'n-' subsamples, calculate based on the absolute n value
                absolute_n = subsample_value  # This is the absolute sample size (e.g., 25 for 'n-25')
                total_n = stats.loc[item, 'n_exp']  # Overall N for the domain or subdomain
                if data_type == 'subdomain':
                    # Format: "Display Label | absolute_n / total_n"
                    x_ticks_labels.append(f'{display_label} | {absolute_n} / {total_n}')
                else:
                    # Format: "Domain\nabsolute_n / total_n"
                    x_ticks_labels.append(f'{item}\n{absolute_n} / {total_n}')
        ax.set_xticks(range(len(order_list)))
        ax.set_xticklabels(x_ticks_labels, rotation=90 if data_type == 'subdomain' else 0, ha='center')
        # Set y-axis limits
        ax.set_ylim([-0.3, 1])
        # Construct the save path based on data_type and fig_number
        if fig_number is not None and idx < len(fig_number):
            prefix = fig_number[idx]  # Get the prefix based on the index
        else:
            prefix = ''  # Default to empty if no prefix or index out of range
        if data_type == 'subdomain':
            fig_filename = f'{prefix}Subdomain_SpatialCorrelations_{subsample}.png'
        else:  # data_type == 'domain'
            fig_filename = f'{prefix}Domain_SpatialCorrelations_{subsample}.png'
        # Construct the full path and save the figure
        if save_dir is not None:
            fig_path = os.path.join(save_dir, fig_filename)
            fig.savefig(fig_path, bbox_inches='tight')

def plot_common_flatmap(
    percent_maps, analysis, granularity, subsampling, 
    order, stats, labels={},
    save_dir=None, figure_number=None, debug=False
):
    """
    Plots the percent maps for each domain (or subdomain) within a specified analysis, granularity, and subsampling setting.
    
    Parameters
    ----------
    percent_maps : dict
        Dictionary of percent maps with the structure percent_maps[analysis][granularity][domain][subsampling].
    analysis : str
        The analysis to process.
    granularity : str
        'domains' or 'subdomains'.
    subsampling : str
        The subsampling setting to plot.
    order : list
        The order in which to plot the domains (or subdomains).
    labels : dict
        translation dictionary from domain names to their print labels
    stats : DataFrame
        A DataFrame containing experiment statistics for each domain.
    save_dir : str, optional
        Directory to save the figure. If None, the figure is not saved.
    figure_number : list of str, optional
        List of prefixes to add to each figure's filename. If None, no prefix is added.
    debug : bool, optional
        If True, prints debug information.
    """
    # Set the figure size and layout based on granularity
    if granularity == 'domains':
        fig = plt.figure(figsize=(40, 6))
        axes_shape = (1, 6)
    else:
        fig = plt.figure(figsize=(40, 30))
        axes_shape = (4, 5)
    # Iterate over each domain/subdomain in the specified order
    for i, domain in enumerate(order):
        # Create subplot dynamically
        ax = fig.add_subplot(*axes_shape, i+1)
        # Check if data is available for the current domain and subsampling
        if domain not in percent_maps[analysis][granularity] or subsampling not in percent_maps[analysis][granularity][domain]:
            if debug:
                print(f"No data available for {domain}. Plotting an empty plot.")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
        else:
            # Plot the actual data
            combined_percent_maps = percent_maps[analysis][granularity][domain][subsampling]
            SUITPy.flatmap.plot(
                combined_percent_maps, 
                colorbar=True, cmap='hot', cscale=(0, 100),
                bordersize=0.5, bordercolor='white', new_figure=False
            )
            # Add a label with the number of experiments
            n_exp_all = stats.loc[domain, 'n_exp'] if domain in stats.index else 'Unknown'
            if subsampling.startswith('n-'):
                n_exp_subsample = int(subsampling.split("-")[1])
            else:
                n_exp_subsample = int(round(float(subsampling.split("-")[1]) / 10 * n_exp_all))
            ax.text(0.5, 0.0, f'(N = {n_exp_subsample} / {n_exp_all} Experiments)', 
                    fontsize=25, ha='center', va='center', transform=ax.transAxes)
        # Set the title for each subplot based on the domain or subdomain
        ax.set_title(f"{labels.get(domain, domain)}", fontsize=60 if granularity == 'domains' else 40)
    # Add overall labels
    fig.text(0.1, 0.5, 'Thresholded Z', va='center', ha='center', rotation='vertical', fontsize=35)
    fig.text(0.12, 0.5, '$\mathit{p}$ < .001 and $\mathit{k}$ = 50', va='center', ha='center', rotation='vertical', fontsize=25)
    # Save the figure if save_dir is specified
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists
        file_number = figure_number[0] if figure_number is not None else ""
        fig_filename = f"{file_number}{analysis}_{granularity}_{subsampling}_Voxel-Heatmap.png"
        fig_path = os.path.join(save_dir, fig_filename)
        fig.savefig(fig_path, bbox_inches='tight')
        plt.close(fig)
    # Show the plot
    plt.show()

def plot_surface(surface_data, mesh, itype=None, filename=None, 
        layout_style='row', cmap='viridis', vrange=None,
        nan_color=(0.75, 0.75, 0.75, 1), **plotter_kwargs):
    """
    Plots `surface_data` on `mesh` using brainspace

    Parameters
    ----------
    surface_data: (np.ndarray)
    mesh: (str | dict)
        - fsaverage
        - fsaverage5
        - dict of path to meshes for 'L' and 'R'
    itype: (str | None)
        mesh file type. For .gii enter None. For freesurfer files enter 'fs'
    filename: (Pathlike str)
    layout_style: (str)
        - row
        - grid
    cmap: (str)
    vrange: (tuple | None)
    nan_color: (tuple)
    **plotter_kwargs
    """
    # create virtual display for plotting in remote servers
    disp=pyvirtualdisplay.Display(visible=False)
    disp.start()
    # load surface mesh
    if isinstance(mesh, str):
        if mesh in ['fsaverage', 'fsaverage5']:
            mesh = {
                'L': nilearn.datasets.fetch_surf_fsaverage(mesh)['infl_left'],
                'R': nilearn.datasets.fetch_surf_fsaverage(mesh)['infl_right'],
            }
            itype = None
        else:
            raise ValueError("Unknown mesh")
    else:
        for fs_suffix in ['.pial', '.midthickness', '.white', '.inflated']:
            if mesh['L'].endswith(fs_suffix):
                itype = 'fs'
    if not os.path.exists(mesh['L']):
        raise ValueError("Mesh not found")
    lh_surf = brainspace.mesh.mesh_io.read_surface(mesh['L'], itype=itype)
    rh_surf = brainspace.mesh.mesh_io.read_surface(mesh['R'], itype=itype)
    # configurations
    if filename:
        screenshot = True
        embed_nb = False
        filename += '.png'
    else:
        screenshot = False
        embed_nb = True
    if layout_style == 'row':
        size = (1600, 400)
        zoom = 1.2
    else:
        size = (900, 500)
        zoom = 1.8
    if vrange is None:        
        vrange = (np.nanmin(surface_data), np.nanmax(surface_data))
    elif vrange == 'sym':
        vmin = min(np.nanmin(surface_data), -np.nanmax(surface_data))
        vrange = (vmin, -vmin)
    return brainspace.plotting.surface_plotting.plot_hemispheres(
        lh_surf, rh_surf, 
        surface_data,
        layout_style = layout_style,
        cmap = cmap, color_range=vrange,
        size=size, zoom=zoom,
        interactive=False, embed_nb=embed_nb,
        screenshot=screenshot, filename=filename, 
        transparent_bg=True,
        nan_color=nan_color,
        **plotter_kwargs
        )

######################
# Data I/O functions #
######################
def load_results(analysis, domain, subdomain=None, flatmap=True, subsamples=None, source='BrainMap'):
    """
    Loads uncorrected and corrected results of the analysis.

    Parameters
    ----------
    analysis: {'ALE', 'SALE'}
        Type of analysis.
    domain: str
        The main domain (BrainMap) or term (Neurosynth) to load results for.
    subdomain: str, optional
        Specific subdomain to load. If not provided, the function will treat the domain as the main result.
        Not applicable to Neurosynth.
    flatmap: bool, optional
        Whether to convert the results to flatmap format (default is True).
    subsamples: str, optional
        The subsamples to load (e.g., 'n-25', 'p-04'). If specified, the function will load the 
        subsamples data instead of the main results.
        Not applicable to Neurosynth.
    source: {'BrainMap', 'Neurosynth'}

    Returns
    -------
    results_dict: dict
        A dictionary containing the loaded z-map results. For subsamples, it will be a nested dictionary 
        with keys [analysis][domain][subsamples] storing lists of z-map results for each subsample.
    """
    if subsamples:
        assert (analysis == 'SALE'), "Subsamples are only available for SALE analysis."
        assert (source == 'BrainMap'), "Subsamples are only available for BrainMap data."
    # Initialize the results dictionary
    results_dict = {}
    # Set the domain path based on whether a subdomain is provided
    if (source == 'BrainMap') and (subdomain is not None):
        domain_full = f"{domain}.{subdomain}"
    else:
        domain_full = domain
    if source == 'BrainMap':
        domain_path = os.path.join(OUTPUT_DIR, analysis, domain, domain_full)
    else:
        domain_path = os.path.join(OUTPUT_DIR, analysis, 'Neurosynth', domain_full.replace(' ', ''))
    # Load mask image for applying cerebellum mask
    mask_img = nibabel.load(create_Diedrichsen2009_mask('MNI', dilation=0))
    # Check if we're loading subsamples
    if subsamples:
        # Initialize a dictionary for subsample results
        results_dict[analysis] = {domain_full: {subsamples: {'uncorr': [], 'corr': []}}}
        # Determine the subsamples path
        subsamples_dir = os.path.join(domain_path, f'subsamples_{subsamples}')
        # Loop over 50 subsample directories with a progress bar
        for i in tqdm(range(50), desc=f"Loading {subsamples} results for {domain}"):
            subsample_path = os.path.join(subsamples_dir, str(i))
            z_uncorr_path = os.path.join(subsample_path, 'uncorr_z.nii.gz')
            # Check if uncorrected results exist for this subsample
            if not os.path.exists(z_uncorr_path):
                print(f"Uncorrected results not found at {z_uncorr_path}. Skipping.")
                continue
            z_uncorr = nibabel.load(z_uncorr_path)
            # Load corrected results for each subsample based on the analysis type
            z_corr_path = os.path.join(subsample_path, 'corr_cluster_h-001_k-50_mask-D2009_MNI_z.nii.gz') # Old-data: 'corr_cluster_k-50_z.nii.gz'

            # Check if corrected results exist for this subsample
            if not os.path.exists(z_corr_path):
                print(f"Corrected results not found at {z_corr_path}. Skipping.")
                continue
            z_corr = nibabel.load(z_corr_path)
            # Apply mask of the cerebellum
            z_uncorr = nilearn.image.math_img("m * img", img=z_uncorr, m=mask_img)
            z_corr = nilearn.image.math_img("m * img", img=z_corr, m=mask_img)
            # Convert to flatmap if indicated
            if flatmap:
                z_uncorr = SUITPy.flatmap.vol_to_surf(z_uncorr, space='SPM').squeeze()
                z_corr = SUITPy.flatmap.vol_to_surf(z_corr, space='SPM').squeeze()
            # Append the results for each subsample
            results_dict[analysis][domain_full][subsamples]['uncorr'].append(z_uncorr)
            results_dict[analysis][domain_full][subsamples]['corr'].append(z_corr)
    else:
        # Load regular (non-subsample) data
        z_uncorr_path = os.path.join(domain_path, 'uncorr_z.nii.gz')
        if not os.path.exists(z_uncorr_path):
            raise FileNotFoundError(f"Uncorrected results not found at {z_uncorr_path}")
        z_uncorr = nibabel.load(z_uncorr_path)
        # Load corrected results based on the analysis type
        if analysis == 'ALE':
            z_corr_path = os.path.join(domain_path, 'cFWE_z_desc-mass_level-cluster_corr-FWE_method-montecarlo.nii.gz')
        elif analysis == 'SALE':
            z_corr_path = os.path.join(domain_path, 'corr_cluster_h-001_k-50_mask-D2009_MNI_z.nii.gz')
        else:
            raise ValueError("Analysis must be 'ALE' or 'SALE'")
        if not os.path.exists(z_corr_path):
            raise FileNotFoundError(f"Corrected results not found at {z_corr_path}")
        z_corr = nibabel.load(z_corr_path)
        # Apply mask of the cerebellum
        z_uncorr = nilearn.image.math_img("m * img", img=z_uncorr, m=mask_img)
        z_corr = nilearn.image.math_img("m * img", img=z_corr, m=mask_img)
        # Convert to flatmap if indicated
        if flatmap:
            z_uncorr = SUITPy.flatmap.vol_to_surf(z_uncorr, space='SPM').squeeze()
            z_corr = SUITPy.flatmap.vol_to_surf(z_corr, space='SPM').squeeze()
        # Store results in the dictionary
        results_dict[analysis] = {domain_full: {'uncorr': z_uncorr, 'corr': z_corr}}
    return results_dict

def load_subsampled_maps(
    domains, output_dir, analysis, subsampling_settings,        
    load_thresholded=False, load_unthresholded=True,
):
    """
    Loads subsampled maps, with options for loading thresholded and/or unthresholded maps,
    and also optionally handles domains and subdomains.

    Parameters
    ----------
    domains : list
        List of domains to load.
    output_dir : str
        Output directory containing the results.
    analysis : str
        Analysis type, e.g., 'SALE'
    subsampling_settings : list
        List of subsampling settings, e.g., ['n-50', 'p-02']. If None, function will exit.
    load_thresholded : bool, optional (default=False)
        If True, load thresholded maps.
    load_unthresholded : bool, optional (default=True)
        If True, load unthresholded maps.

    Returns
    -------
    tuple
        (surf_zmaps, surf_zmaps_thr) where each is a dictionary 
        containing the loaded maps. Returns (None, None) if no maps are loaded.
    """
    # Check that at least one map type is to be loaded
    if not load_thresholded and not load_unthresholded:
        print("Both load_thresholded and load_unthresholded are False. No maps will be loaded.")
        return None, None
    surf_zmaps = {}
    surf_zmaps_thr = {}
    surf_zmaps[analysis] = {}
    surf_zmaps_thr[analysis] = {}
    for domain in domains:
        current_key = domain
        domain_parent = domain.split('.')[0]
        subdomain = None
        if '.' in domain:
            subdomain = '.'.join(domain.split('.')[1:])
        path_part = os.path.join(domain_parent, domain)
        # Initialize current domain entry
        surf_zmaps[analysis][current_key] = {}
        if load_thresholded:
            surf_zmaps_thr[analysis][current_key] = {}
        print(f"Loading {analysis} results for domain {current_key}")
        for subsampling in tqdm(subsampling_settings, desc=f"Processing {current_key}"):
            # Check if the subsampling folder exists before loading results
            subsample_path = os.path.join(output_dir, analysis, path_part, f'subsamples_{subsampling}')
            if not os.path.exists(subsample_path):
                print(f"Subsampling folder not found: {subsample_path}. Skipping...")
                continue
            try:
                # Load results for subsamples
                results = load_results(analysis, domain_parent, subdomain=subdomain, subsamples=subsampling)
            except FileNotFoundError:
                print(f"Results not found for {analysis} in {current_key} with subsampling {subsampling}. Skipping...")
                continue                 
            # Initialize lists for each subsampling type
            surf_zmaps[analysis][current_key][subsampling] = []
            if load_thresholded:
                surf_zmaps_thr[analysis][current_key][subsampling] = []
            # Append each subsample's results to the lists
            for z_uncorr, z_corr in zip(results[analysis][domain][subsampling]['uncorr'],
                                        results[analysis][domain][subsampling]['corr']):
                if load_unthresholded:
                    surf_zmaps[analysis][current_key][subsampling].append(np.array(z_uncorr))
                if load_thresholded:
                    surf_zmaps_thr[analysis][current_key][subsampling].append(np.array(z_corr))
            # Convert lists to arrays after loading all subsamples
            if load_unthresholded:
                surf_zmaps[analysis][current_key][subsampling] = np.array(surf_zmaps[analysis][current_key][subsampling])
            if load_thresholded:
                surf_zmaps_thr[analysis][current_key][subsampling] = np.array(surf_zmaps_thr[analysis][current_key][subsampling])
    # Return the dictionaries with the loaded data
    return (surf_zmaps if load_unthresholded else None, 
            surf_zmaps_thr if load_thresholded else None)

###############
# Misc. utils #
###############
def seconds_to_str(seconds):
    """
    Converts seconds to "??s", "??m??s", "??h??m" or "??d"

    Parameters
    ----------
    seconds : float

    Returns
    -------
    seconds_str: str
    """
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes, rem_seconds = divmod(seconds, 60)
        return f"{int(minutes)}m{int(rem_seconds)}s"
    elif seconds < 86400:
        hours, rem_seconds = divmod(seconds, 3600)
        rem_minutes = rem_seconds // 60
        return f"{int(hours)}h {int(rem_minutes)}m"
    else:
        days = seconds // 86400
        return f"{int(days)}d"