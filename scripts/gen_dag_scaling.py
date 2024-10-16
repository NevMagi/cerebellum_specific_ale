import os

def gen_dag():
    os.makedirs('./dag', exist_ok=True)
    dag_filename = f'./dag/run_scaling.dag'
    f = open('/data/project/cerebellum_ale/scripts/'+dag_filename, 'w')
    submit_file = '/data/project/cerebellum_ale/scripts/run_scaling.submit'
    job_count = 0
    for analysis in ['ale', 'dsale']:
        # scaling of number of experiments with fixed n_iters=100
        for n_exp in range(100, 1100, 100):
            f.write(f'JOB job_{job_count} {submit_file}\n')
            f.write(f'VARS job_{job_count} analysis="{analysis}" n_iters="100" n_exp="{n_exp}"\n\n')
            job_count += 1
        # scaling of number of iterations with fixed experiments=100
        for n_iters in [10, 100, 1000, 10000]:
            f.write(f'JOB job_{job_count} {submit_file}\n')
            f.write(f'VARS job_{job_count} analysis="{analysis}" n_iters="{n_iters}" n_exp="100"\n\n')
            job_count += 1
    f.write(f'\nCATEGORY ALL_NODES run_scaling')
    f.write(f'\nMAXJOBS run_scaling 100\n\n')

if __name__ == '__main__':
    gen_dag()