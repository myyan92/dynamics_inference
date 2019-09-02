from redis_client import RedisQueue, RedisHash
import json
from physbam_python.rollout_physbam_3d import rollout_single as rollout_single_3d
import numpy as np
import signal
from multiprocessing import Process, Value, cpu_count
from ctypes import c_bool
import pdb

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
                state = rollout_single_3d(**task_dict)
                return_str = json.dumps({'state':state.tolist()})
                result_hash.put(job_id, return_str)
    except KeyboardInterrupt:
        print('receive ctrl-c')
        terminate.value=True

if __name__=="__main__":
    num_workers = cpu_count() # - 4
    workers = []
    # TODO add lock in queue and hash?
    task_queue = RedisQueue()
    task_queue.clear()
    state = np.zeros((64,3))
    state[:,0]=np.linspace(0,1,64)
    action = np.zeros((10,4))
    action[:,1] = 0.02
    physbam_args = ' -friction 0.1 -self_friction 0.1 -stiffen_linear 1.0 -stiffen_bending 1.0'
    for i in range(4):
        action[:,3]=i*0.1+0.1
        job_id = 'JOB%d'%(i)
        task_str = json.dumps({'state':state.tolist(),'action':action.tolist(),'physbam_args':physbam_args,'job_id':job_id})
        task_queue.put(task_str)

    result_hash = RedisHash()
    result_hash.clear()
    terminate = Value(c_bool, False)
    for _ in range(num_workers):
        workers.append(Process(target=process_func, args=(task_queue, result_hash, terminate)))
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
