import os
import glob

# define input and output directories
INPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'input')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')

def generate_dagman_file():
    """
    Generates a .dagman file for running the variogram jobs using HTCondor
    """
    # make dag directory
    dag_dir = os.path.join(os.path.dirname(__file__), 'dag')
    os.makedirs(dag_dir, exist_ok=True)
    # determine dagman file name
    dag_path = os.path.join(dag_dir, 'run_variogram.dag')
    f = open(dag_path, 'w')
    # set submit file
    submit_file = os.path.join(os.path.dirname(__file__), 'run_variogram.submit')
    # list all SALE unthresholded z-maps
    zmap_paths = glob.glob(os.path.join(OUTPUT_DIR, 'SALE', '*', '*', 'uncorr_z.nii.gz'))
    for i, zmap_path in enumerate(zmap_paths):
        if 'Neurosynth' in zmap_path:
            # skip Neurosynth z-maps
            continue
        f.write(f'JOB job_{i} {submit_file}\n')
        f.write(f'VARS job_{i} zmap_path="{zmap_path}"\n\n')
    maxjobs = 50 # number of ongoing jobs at a time
    f.write('\nCATEGORY ALL_NODES variogram')
    f.write(f'\nMAXJOBS variogram {maxjobs}\n\n')


if __name__ == '__main__':
    generate_dagman_file()
