"""
Generates a .dagman file including 3000 jobs (100 hyperparameter sets * 30 folds)
"""
import sys
import pandas as pd

MIN_EXPERIMENTS = 15

def generate_dagman_file(analysis, subsampling, subsample_subbds=False):
    prefix = f'run_{analysis}'
    if subsampling == 1:
        prefix += '_subsampling'
    elif subsampling == 2:
        prefix += '_subsampling_p'
    if subsample_subbds:
        prefix += '_subbds'
    if analysis == 'variogram':
        submit_file = '/data/project/cerebellum_ale/scripts/run_cpu.submit'
        maxjobs = 50
    else:
        submit_file = '/data/project/cerebellum_ale/scripts/run.submit'
        maxjobs = 5
    dag_filename = prefix + '.dag'
    f = open('/data/project/cerebellum_ale/scripts/'+dag_filename, 'w')
    dsets = pd.read_csv('/data/project/cerebellum_ale/output/exp_stats_240209.csv', 
                        index_col=0).sort_values('n_experiments', ascending=False)
    dsets = dsets[dsets['n_experiments'] >= MIN_EXPERIMENTS]
    n_subsamples = 0
    subsample_size = 0
    if subsampling > 0:
        # set n and size of subsamples
        n_subsamples = 50
        if subsampling == 1:
            subsample_size = 50
        elif subsampling == 2:
            subsample_size = 0.2
        if subsample_subbds:
            dsets = dsets[
                (dsets['n_experiments'] >= 100) & \
                (dsets['SubBD'] != dsets['BD'])
            ]
        else:
            dsets = dsets[dsets['SubBD'] == dsets['BD']]
    job_count = 0
    for _, row in dsets.iterrows():
        bd = row['BD']
        subbd = row['SubBD']
        # replace illegal characters in subbd names
        subbd = subbd.replace(' ', '_space_').replace('/', '_slash_')
        f.write(f'JOB job_{job_count} {submit_file}\n')
        f.write(f'VARS job_{job_count} analysis="{analysis}" bd="{bd}" subbd="{subbd}" n_subsamples="{n_subsamples}" subsample_size="{subsample_size}"\n\n')
        job_count += 1
    f.write(f'\nCATEGORY ALL_NODES {prefix}')
    f.write(f'\nMAXJOBS {prefix} {maxjobs}\n\n') # keep three GPUs free


if __name__ == '__main__':
    analysis = sys.argv[1]
    subsampling = int(sys.argv[2])
    try:
        subsample_subbds = bool(int(sys.argv[3]))
    except:
        subsample_subbds = False
    generate_dagman_file(analysis, subsampling, subsample_subbds)
