"""
Generates a .dagman file including 3000 jobs (100 hyperparameter sets * 30 folds)
"""
import sys
import pandas as pd

MIN_EXPERIMENTS = 15

def generate_dagman_file(analysis, subsampling):
    prefix = f'run_{analysis}'
    if subsampling:
        prefix += '_subsampling'
    dag_filename = prefix + '.dag'
    f = open('/data/project/cerebellum_ale/scripts/'+dag_filename, 'w')
    dsets = pd.read_csv('/data/project/cerebellum_ale/output/exp_stats.csv').sort_values('n_experiments', ascending=False)
    dsets = dsets[dsets['n_experiments'] >= MIN_EXPERIMENTS]
    n_subsamples = 0
    if subsampling:
        n_subsamples = 100
        # for now limit them to 'All' subbds
        dsets = dsets[dsets['SubBD'] == 'All']
        # # in case of subsampling as subsample size is 20
        # # we need to remove datasets with less than 30 experiments
        # dsets = dsets[dsets['n_experiments'] >= 30]
    job_count = 0
    for _, row in dsets.iterrows():
        bd = row['BD']
        subbd = row['SubBD']
        subbd = subbd.replace(" - ", "_") # replace illegal " - " with "_", will be replaced back in run.py
        if subbd == 'Social Cognition': # quick fix for social cognition
            subbd = 'SocialCognition'
        f.write(f'JOB job_{job_count} run.submit\n')
        f.write(f'VARS job_{job_count} analysis="{analysis}" bd="{bd}" subbd="{subbd}" n_subsamples="{n_subsamples}"\n\n')
        job_count += 1
    f.write(f'\nCATEGORY ALL_NODES {prefix}')
    f.write(f'\nMAXJOBS {prefix} 5\n\n') # keep three GPUs free


if __name__ == '__main__':
    analysis = sys.argv[1]
    subsampling = bool(int(sys.argv[2]))
    generate_dagman_file(analysis, subsampling)