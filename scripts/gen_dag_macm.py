import os
import glob
import nibabel
import numpy as np

# define input and output directories
INPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'input')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')

def generate_dagman_file():
    """
    Generates a .dagman file for running the MACM jobs using HTCondor
    """
    # make dag directory
    dag_dir = os.path.join(os.path.dirname(__file__), 'dag')
    os.makedirs(dag_dir, exist_ok=True)
    dag_path = os.path.join(dag_dir, 'run_macm.dag')
    f = open(dag_path, 'w')
    # set submit file
    submit_file = os.path.join(os.path.dirname(__file__), 'run_macm.submit')
    # list all SALE thresholded and cluster-extent-corrected z-maps
    zmap_paths = glob.glob(os.path.join(OUTPUT_DIR, 'SALE', '*', '*', 'corr_cluster_h-001_k-50_mask-D2009_MNI_z.nii.gz'))
    job_count = 0
    for zmap_path in zmap_paths:
        zmap = nibabel.load(zmap_path)
        # only run MACM if there are significant clusters
        sig = np.any(zmap.get_fdata().flatten()>0)
        if sig:
            f.write(f'JOB job_{job_count} {submit_file}\n')
            f.write(f'VARS job_{job_count} zmap_path="{zmap_path}"\n\n')
            job_count += 1
    maxjobs = 5 # number of ongoing jobs at a time
    f.write('\nCATEGORY ALL_NODES macm')
    f.write(f'\nMAXJOBS macm {maxjobs}\n\n')


if __name__ == '__main__':
   generate_dagman_file()
