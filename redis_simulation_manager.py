from redis_client import RedisQueue, RedisHash
import json
from physbam_python.rollout_physbam_3d import rollout_single as rollout_single_3d
import numpy as np
import signal, sys, os, shutil
from multiprocessing import Process, Value, cpu_count
from ctypes import c_bool

def process_func(task_queue, result_hash, terminate):
    try:
        while not terminate.value:
            task_str = task_queue.get(timeout=1)
            if task_str is not None:
                task_str = task_str.decode('utf-8')
                task_dict = json.loads(task_str)
                job_id = task_dict.pop('job_id')
                task_dict['state']=np.array(task_dict['state'])
                task_dict['action']=np.array(task_dict['action'])
                task_dict['input_raw']=True
                task_dict['return_raw']=True,
                reset_spring = task_dict.pop('reset_spring')
                idx = task_dict.pop('idx')
                if not reset_spring:
                    if (not os.path.exists(os.path.join('./physbam_3d_springs_tmp', 'linear_%02d.txt'%(idx)))) or \
                       (not os.path.exists(os.path.join('./physbam_3d_springs_tmp', 'bending_%02d.txt'%(idx)))):
                        raise ValueError('no spring files saved')
                task_dict['save_linear_spring']=os.path.join('./physbam_3d_springs_tmp', 'linear_%02d.txt'%(idx)) if reset_spring else None
                task_dict['save_bending_spring']=os.path.join('./physbam_3d_springs_tmp', 'bending_%02d.txt'%(idx)) if reset_spring else None
                task_dict['load_linear_spring']=os.path.join('./physbam_3d_springs_tmp', 'linear_%02d.txt'%(idx)) if not reset_spring else None
                task_dict['load_bending_spring']=os.path.join('./physbam_3d_springs_tmp', 'bending_%02d.txt'%(idx)) if not reset_spring else None

                state = rollout_single_3d(**task_dict)
                if state is None:
                    print('job id ', job_id, ' failed')
                    return_str = json.dumps({'state':None})
                else:
                    print('job id ', job_id, ' finished')
                    return_str = json.dumps({'state':state.tolist()})
                result_hash.put(job_id, return_str)
    except KeyboardInterrupt:
        print('receive ctrl-c')
        terminate.value=True

if __name__=="__main__":
    redis_host = sys.argv[1]
    num_workers = cpu_count() - 4
    workers = []
    task_queue = RedisQueue(host=redis_host)
    result_hash = RedisHash(host=redis_host)
    result_hash.clear()
    terminate = Value(c_bool, False)

    if os.path.isdir('./physbam_3d_springs_tmp'):
        shutil.rmtree('./physbam_3d_springs_tmp')
    os.mkdir('./physbam_3d_springs_tmp')

    for _ in range(num_workers):
        workers.append(Process(target=process_func, args=(task_queue, result_hash, terminate)))
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
