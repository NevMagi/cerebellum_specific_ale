import os
import sys

def gen_dag(use_gpu):
    """
    Generates a .dagman file for running the scaling jobs using HTCondor

    Parameters
    ----------
    use_gpu : bool
    """
    # make dag directory
    dag_dir = os.path.join(os.path.dirname(__file__), 'dag')
    os.makedirs(dag_dir, exist_ok=True)
    # determine dagman file name, submit file and analyses depending on use_gpu
    prefix = 'run_scaling'
    if use_gpu:
        prefix += '_gpu'
        submit_file = os.path.join(os.path.dirname(__file__), 'run_scaling_gpu.submit')
        analyses = ['ale', 'dsale', 'sale']
    else:
        prefix += '_cpu'
        submit_file = os.path.join(os.path.dirname(__file__), 'run_scaling_cpu.submit')
        analyses = ['ale', 'dsale'] # psale not implemented on CPU
    dag_path = os.path.join(dag_dir, f'{prefix}.dag')
    f = open(dag_path, 'w')
    # write jobs to dag file
    job_count = 0
    for analysis in analyses:
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
    # set maxjobs
    if use_gpu:
        max_jobs = 5
    else:
        max_jobs = 100
    f.write(f'\nCATEGORY ALL_NODES {prefix}')
    f.write(f'\nMAXJOBS {prefix} {max_jobs}\n\n')

if __name__ == '__main__':
    use_gpu = bool(int(sys.argv[1]))
    gen_dag(use_gpu)