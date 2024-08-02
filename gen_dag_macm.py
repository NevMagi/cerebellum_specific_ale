import sys
import pandas as pd
import glob
import nibabel
import numpy as np

def generate_dagman_file():
    dag_filename = './dag/run_macm.dag'
    f = open('/data/project/cerebellum_ale/scripts/'+dag_filename, 'w')
    zmap_paths = glob.glob('/data/project/cerebellum_ale/output/SALE/*/*/corr_cluster_h-001_k-50_mask-D2009_MNI_z.nii.gz')
    job_count = 0
    for zmap_path in zmap_paths:
        zmap = nibabel.load(zmap_path)
        sig = np.any(zmap.get_fdata().flatten()>0)
        if sig:
            f.write(f'JOB job_{job_count} run_macm.submit\n')
            f.write(f'VARS job_{job_count} zmap_path="{zmap_path}"\n\n')
            job_count += 1
    f.write(f'\nCATEGORY ALL_NODES macm')
    f.write(f'\nMAXJOBS macm 5\n\n') # keep three GPUs free


if __name__ == '__main__':
   generate_dagman_file()
