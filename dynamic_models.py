import numpy as np
import os, shutil
import gin
import json
import os, glob
import time
from multiprocessing import Pool
from physbam_python.rollout_physbam_2d import rollout_single as rollout_single_2d
from physbam_python.rollout_physbam_3d import rollout_single as rollout_single_3d

GROUP_SCRATCH = os.environ.get('GROUP_SCRATCH')
SLURM_JOB_ID_MASTER = os.environ.get('SLURM_JOB_ID_PACK_GROUP_0', '')

def _pickle_method(method):
	func_name = method.im_func.__name__
	obj = method.im_self
	cls = method.im_class
	if func_name.startswith('__') and not func_name.endswith('__'): #deal with mangled names
		cls_name = cls.__name__.lstrip('_')
		func_name = '_' + cls_name + func_name
	return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
	for cls in cls.__mro__:
		try:
			func = cls.__dict__[func_name]
		except KeyError:
			pass
		else:
			break
	return func.__get__(obj, cls)

import copyreg
import types
copyreg.pickle(types.MethodType, _pickle_method, _unpickle_method)


@gin.configurable
class physbam_2d(object):
  def __init__(self, physbam_args=" -disable_collisions -stiffen_bending 100"):
    self.physbam_args = physbam_args

  def execute(self, state, actions, **kwargs):
    """Execute action sequence and get end state.
    Args:
      state: (N,2) array representing the current state.
      actions: A list of (Int, array) tuple, where the int is
          the action node, and the array is 2d vector of action.
    """
    for ac in actions:
      state = rollout_single_2d(state, ac[0]+1, ac[1], frames=1,
                               physbam_args=self.physbam_args)
    return state

  def execute_batch(self, state, actions, **kwargs):
    """Execute in batch.
    Args:
      state: (N,2) array or a list of (N,2) array representing the current state.
      actions: A list of list of (Int, array) tuple.
    """
    if isinstance(state, list):
        assert(len(state)==len(actions))
    else:
        assert(state.ndim==2)
        state = [state for a in actions]
    pool = Pool()
    states = pool.starmap(self.execute, zip(state, actions))
    pool.close()
    pool.join()
    return states


@gin.configurable
class physbam_3d(object):
  def __init__(self, physbam_args=" -friction 0.176 -stiffen_linear 2.223 -stiffen_bending 0.218"):
    self.physbam_args = physbam_args
    if os.path.isdir(os.path.join(GROUP_SCRATCH, 'mengyuan/rope_manipulation', SLURM_JOB_ID_MASTER+'physbam_3d_springs_tmp')):
        shutil.rmtree(os.path.join(GROUP_SCRATCH, 'mengyuan/rope_manipulation', SLURM_JOB_ID_MASTER+'physbam_3d_springs_tmp'))
    os.mkdir(os.path.join(GROUP_SCRATCH, 'mengyuan/rope_manipulation', SLURM_JOB_ID_MASTER+'physbam_3d_springs_tmp'))

    files = glob.glob(os.path.join(GROUP_SCRATCH, 'mengyuan/rope_manipulation', SLURM_JOB_ID_MASTER+'task_*.json'))
    for f in files:
        os.remove(f)
    files = glob.glob(os.path.join(GROUP_SCRATCH, 'mengyuan/rope_manipulation', SLURM_JOB_ID_MASTER+'result_*.npy'))
    for f in files:
        os.remove(f)

  def execute(self, state, actions, return_traj=False, reset_spring=True, idx=0):
    """Execute action sequence and get end state.
    Args:
      current: (N,2) array representing the current state.
      actions: A list of (Int, array) tuple, where the int is
          the action node, and the array is 2d/3d vector of action.
    """
    # transform actions
    moves = np.array([ac[1] for ac in actions])
    nodes = np.array([[float(ac[0])/(state.shape[0]//2-1)] for ac in actions])
    actions = np.concatenate([moves, nodes],axis=1)
    assert(state.shape==(128,3)) # must be raw state
    if not reset_spring:
        if (not os.path.exists(os.path.join('./physbam_3d_springs_tmp', 'linear_%02d.txt'%(idx)))) or \
           (not os.path.exists(os.path.join('./physbam_3d_springs_tmp', 'bending_%02d.txt'%(idx)))):
            raise ValueError('no spring files saved')
    with open(os.path.join(GROUP_SCRATCH, 'mengyuan/rope_manipulation', SLURM_JOB_ID_MASTER+'task_%03d_tmp.json'%(id)), 'w') as f:
        json.dump({'state':state.tolist(),'action':actions.tolist(), 'physbam_args':' -dt 1e-3 ' + self.physbam_args,
                   'return_traj':return_traj, 'input_raw':True, 'return_raw':True, 'reset_spring':reset_spring, 'idx':idx}, f)
    os.rename( os.path.join(GROUP_SCRATCH, 'mengyuan/rope_manipulation', SLURM_JOB_ID_MASTER+'task_%03d_tmp.json'%(id)),
               os.path.join(GROUP_SCRATCH, 'mengyuan/rope_manipulation', SLURM_JOB_ID_MASTER+'task_%03d.json'%(id)) )
    print("job %d posted"%(id), time.time())
    loaded_result = False
    while not loaded_result:
        try:
            state = np.load( os.path.join(GROUP_SCRATCH, 'mengyuan/rope_manipulation', SLURM_JOB_ID_MASTER+'result_%03d.npy'%(id)))
            loaded_result=True
            os.remove( os.path.join(GROUP_SCRATCH, 'mengyuan/rope_manipulation', SLURM_JOB_ID_MASTER+'result_%03d.npy'%(id)))
        except:
            time.sleep(0.1)
    print("job %d received"%(id), time.time())
    if state.shape == (): # np.array(None)
        return None
    return state

  def execute_batch(self, state, actions, return_traj=False, reset_spring=True):
    """Execute in batch.
    Args:
      state: (N,2) array or a list of (N,2) array representing the current state.
      actions: A list of list of (Int, array) tuple.
    """
    if isinstance(state, list):
        assert(len(state)==len(actions))
    else:
        assert(state.ndim==2)
        state = [state for a in actions]

    for id, (st, ac) in enumerate(zip(state, actions)):
        moves = np.array([a[1] for a in ac])
        nodes = np.array([[float(a[0])/(st.shape[0]//2-1)] for a in ac])
        actions = np.concatenate([moves, nodes],axis=1)
        assert(st.shape==(128,3)) # must be raw state
        with open(os.path.join(GROUP_SCRATCH, 'mengyuan/rope_manipulation', SLURM_JOB_ID_MASTER+'task_%03d_tmp.json'%(id)), 'w') as f:
            json.dump({'state':st.tolist(),'action':actions.tolist(), 'physbam_args':' -dt 1e-3 ' + self.physbam_args,
                       'return_traj':return_traj, 'input_raw':True, 'return_raw':True, 'reset_spring':reset_spring, 'idx':id}, f)
        os.rename( os.path.join(GROUP_SCRATCH, 'mengyuan/rope_manipulation', SLURM_JOB_ID_MASTER+'task_%03d_tmp.json'%(id)),
                   os.path.join(GROUP_SCRATCH, 'mengyuan/rope_manipulation', SLURM_JOB_ID_MASTER+'task_%03d.json'%(id)) )
        #print("job %d posted"%(id), time.time())

    states = [None]*len(state)
    loaded_result = [False]*len(state)
    while not np.all(loaded_result):
        for id in range(len(states)):
            debug_path = os.path.join(GROUP_SCRATCH, 'mengyuan/rope_manipulation', SLURM_JOB_ID_MASTER+'result_%03d.npy'%(id))
            if os.path.exists(os.path.join(GROUP_SCRATCH, 'mengyuan/rope_manipulation', SLURM_JOB_ID_MASTER+'result_%03d.npy'%(id))):
                loaded_result[id] = True
        time.sleep(2)
    time.sleep(2)
    for id in range(len(states)):
        states[id] = np.load( os.path.join(GROUP_SCRATCH, 'mengyuan/rope_manipulation', SLURM_JOB_ID_MASTER+'result_%03d.npy'%(id)), allow_pickle=True)
        os.remove( os.path.join(GROUP_SCRATCH, 'mengyuan/rope_manipulation', SLURM_JOB_ID_MASTER+'result_%03d.npy'%(id)))

    for id,st in enumerate(states):
        if st.shape == (): # np.array(None)
            states[id] = None
    return states

  def close(self):
    shutil.rmtree('./physbam_3d_springs_tmp')


if __name__=='__main__':
    sc = physbam_2d()
    state = np.zeros((64,2))
    actions = [[( 0,np.zeros((2,)) )]]
    sc.execute_batch(state, actions)
