import numpy as np
import os, shutil
import gin
from multiprocessing import Pool
from physbam_python.rollout_physbam_2d import rollout_single as rollout_single_2d
from physbam_python.rollout_physbam_3d import rollout_single as rollout_single_3d

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
    if os.path.isdir('./physbam_3d_springs_tmp'):
        shutil.rmtree('./physbam_3d_springs_tmp')
    os.mkdir('./physbam_3d_springs_tmp')

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
    state = rollout_single_3d(state, actions, physbam_args=' -dt 1e-3 ' + self.physbam_args,
                              return_traj=return_traj, input_raw=True, return_raw=True,
                              save_linear_spring=os.path.join('./physbam_3d_springs_tmp', 'linear_%02d.txt'%(idx)) if reset_spring else None,
                              save_bending_spring=os.path.join('./physbam_3d_springs_tmp', 'bending_%02d.txt'%(idx)) if reset_spring else None,
                              load_linear_spring=os.path.join('./physbam_3d_springs_tmp', 'linear_%02d.txt'%(idx)) if not reset_spring else None,
                              load_bending_spring=os.path.join('./physbam_3d_springs_tmp', 'bending_%02d.txt'%(idx)) if not reset_spring else None
                              )
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
    return_traj = [return_traj for a in actions]
    reset_spring = [reset_spring for a in actions]
    idx = [i for i,a in enumerate(actions)]
    pool = Pool()
    states = pool.starmap(self.execute, zip(state, actions, return_traj, reset_spring, idx))
    pool.close()
    pool.join()
    return states

  def close(self):
    shutil.rmtree('./physbam_3d_springs_tmp')


import tensorflow as tf
from neural_simulator.model_wrapper import Model


@gin.configurable
class neural_sim(object):
  def __init__(self, model_type, snapshot):
    self.start = tf.placeholder(tf.float32, shape=[None, 64, 3])
    self.action = tf.placeholder(tf.float32, shape=[None, 64, 5])

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth=True
    self.sess = tf.Session(config=tf_config)
    self.model = Model(model_type)
    self.model.build(input=self.start, action=self.action)
    self.sess.run(tf.global_variables_initializer())
    self.model.load(self.sess, snapshot)
    print("Warning: This model is hacked to work with state and action in robot coordinate.")
    print("It will convert input state and action to fit the distribution of training data")

  def execute(self, state, actions, **kwargs):
#    state = state*np.array([-1.0, 1.0])+np.array([0.5, 0.0])
#    actions = [(a[0], np.array([-a[1][0], a[1][1]])) for a in actions]

    onehot_actions = []
    for ac in actions:
      onehot_action = np.zeros((state.shape[0], ac[1].shape[0]))
      onehot_action[ac[0],:] = ac[1]
      onehot_actions.append(onehot_action)
    for ac in onehot_actions:
      state = self.model.predict_single(self.sess, state, ac)
 #   state[:,0]=0.5-state[:,0]
    return state

  def execute_batch(self, state, actions, **kwargs):
    if isinstance(state, list) or state.ndim==3:
        assert(len(state)==len(actions))
    else:
        assert(state.ndim==2)
        state = [state for a in actions]
    state = np.array(state)

#    state = state*np.array([-1.0, 1.0])+np.array([0.5, 0.0])
#    actions = [[(a[0], np.array([-a[1][0], a[1][1]])) for a in action] for action in actions]

    num_steps = [len(action) for action in actions]
    max_num_steps = np.amax(num_steps)

    final_state = np.zeros_like(state)
    for t in range(max_num_steps):
      onehot_ac = np.zeros((state.shape[0], state.shape[1], actions[0][0][1].shape[0]+2))
      for i,action in enumerate(actions):
        if t<num_steps[i]:
          onehot_ac[i,action[t][0],:-2] = action[t][1]
          # compute action angle feature.
          start_ori = state[i, min(action[t][0]+1,state.shape[1]-1)] - state[i, max(action[t][0]-1,0)]
          start_theta = np.arctan2(start_ori[1], start_ori[0])
          if np.linalg.norm(action[t][1][:2]) > 3e-3:
            target_theta = np.arctan2(action[t][1][1],action[t][1][0])
            if action[t][0] <= 4:
              target_theta += np.pi/2.0
            elif action[t][0] > 59:
              target_theta -= np.pi/2.0
            elif np.dot(action[t][1][:2], start_ori[:2]) < 0.0:
              target_theta += np.pi
          else:
            target_theta=start_theta

          act_angle = np.array([np.sin(target_theta-start_theta), np.cos(target_theta-start_theta)])
          onehot_ac[i,action[t][0],-2:] = act_angle
      state = self.model.predict_batch(self.sess, state, onehot_ac)
      for i,st in enumerate(state):
        if t==num_steps[i]-1:
          final_state[i] = st
#    state[:,:,0]=0.5-state[:,:,0]
    return final_state


if __name__=='__main__':
    sc = physbam_2d()
    state = np.zeros((64,2))
    actions = [[( 0,np.zeros((2,)) )]]
    sc.execute_batch(state, actions)
