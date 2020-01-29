import json
from physbam_python.rollout_physbam_3d import rollout_single as rollout_single_3d
import numpy as np
import signal, sys, os
import time
from multiprocessing import Process

GROUP_SCRATCH = os.environ.get('GROUP_SCRATCH')
NUM_WORKERS = os.environ.get('SLURM_CPUS_PER_TASK')
SLURM_JOB_ID_MASTER = os.environ.get('SLURM_JOB_ID_PACK_GROUP_0','')

def process_func(task_file, result_file):
    try:
        while True:
            loaded_task = False
            while not loaded_task:
                try:
                    with open(task_file) as f:
                        task_dict = json.load(f)
                    loaded_task = True
                    os.remove(task_file)
                except:
                    time.sleep(0.1)
            task_dict['state']=np.array(task_dict['state'])
            task_dict['action']=np.array(task_dict['action'])
            reset_spring = task_dict.pop('reset_spring')
            idx = task_dict.pop('idx')
            task_dict['save_linear_spring'] = os.path.join(GROUP_SCRATCH, 'mengyuan/rope_manipulation',
                              SLURM_JOB_ID_MASTER+'physbam_3d_springs_tmp', 'linear_%02d.txt'%(idx)) if reset_spring else None
            task_dict['save_bending_spring'] = os.path.join(GROUP_SCRATCH, 'mengyuan/rope_manipulation',
                              SLURM_JOB_ID_MASTER+'physbam_3d_springs_tmp', 'bending_%02d.txt'%(idx)) if reset_spring else None
            task_dict['load_linear_spring'] = os.path.join(GROUP_SCRATCH, 'mengyuan/rope_manipulation',
                              SLURM_JOB_ID_MASTER+'physbam_3d_springs_tmp', 'linear_%02d.txt'%(idx)) if not reset_spring else None
            task_dict['load_bending_spring'] = os.path.join(GROUP_SCRATCH, 'mengyuan/rope_manipulation',
                              SLURM_JOB_ID_MASTER+'physbam_3d_springs_tmp', 'bending_%02d.txt'%(idx)) if not reset_spring else None

            state = rollout_single_3d(**task_dict)
            if state is None:
                print('job id ', task_file, ' failed')
            np.save(result_file.replace('.npy', '_tmp.npy'), state)
            os.rename(result_file.replace('.npy', '_tmp.npy'), result_file)
    except KeyboardInterrupt:
        print('receive ctrl-c')

if __name__=="__main__":
    task_id_offset = int(sys.argv[1])
    num_workers = int(NUM_WORKERS)
    workers = []
    for id in range(num_workers):
        task_file = os.path.join(GROUP_SCRATCH, 'mengyuan/rope_manipulation', SLURM_JOB_ID_MASTER+'task_%03d.json'%(id + task_id_offset))
        result_file = os.path.join(GROUP_SCRATCH, 'mengyuan/rope_manipulation', SLURM_JOB_ID_MASTER+'result_%03d.npy'%(id + task_id_offset))
        workers.append(Process(target=process_func, args=(task_file, result_file)))
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
