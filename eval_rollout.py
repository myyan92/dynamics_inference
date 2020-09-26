import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os, pdb, time, glob
from PIL import Image
import argparse, gin
from visual_inference.hybrid_inference import HybridInference
from TF_cloth2d.models.model_VGG_STN_2 import Model_STNv2
from dynamics_inference.dynamic_models import *
from physbam_python.state_to_mesh import state_to_mesh

def load_sim_sequence(dir, use_vision=False):
    index = dir.split('/')[-1]
    with open(os.path.join(dir, index+'_act.txt')) as f:
        lines = f.readlines()
    actions = [l.strip().split() for l in lines]
    actions = [(int(l[0]), float(l[1]), float(l[2])) for l in actions]
    actions = [(ac[0], np.array([ac[1], ac[2]])) for ac in actions]
    num_actions = len(actions)
    states = []
    for i in range(num_actions+1):
        state = np.loadtxt(os.path.join(dir, index+'_%03d.txt'%(i)))
        state[:,1]=-state[:,1] # for seq3d data
        states.append(state)
    if use_vision:
        im = Image.open(os.path.join(dir, index+'_001.png'))
        image = np.array(im)[::-1,:,:] # for seq3d data
        inferencer.fitting.init_position=None # reset memory to be safe
        input_state = inferencer.inference(image)
        # convert frame
        # align to true state
    else:
        input_state = states[1]
    return input_state, states[1:], actions[1:]

def load_real_sequence_old(dir):
    inferencer.fitting.init_position=None # reset memory
    actions = np.load(os.path.join(dir, 'actions.npy'))
    actions = actions[:,[12,13,16,17]]
    num_actions = len(actions)
    states = []
    for i in range(num_actions+1):
        image = Image.open(os.path.join(dir, 'image_%d.png'%(i)))
        state = inferencer.inference(np.array(image)) # need to set memory to true
        states.append(state)
        # convert frame

    converted_actions = []
    for act,state  in zip(actions, states):
        dists = np.linalg.norm(state-act[:2], axis=-1)
        action_node = np.argmin(dists)
        action = (action_node, np.array([act[2], act[3]]))
        converted_actions.append(action)

    return states[0], states, converted_actions

def load_real_sequence_new(dir):
    # for real_rope_with_occlusion-new
    data = np.load(os.path.join(dir, 'history_perception.npz'))
    states = data['perception']
    actions = data['actions'] # array of objects

    converted_actions = []
    fully_observe = []
    action_node = None
    start_position = None
    for act, state in zip(actions, states):
        if len(act)==4:
            dists = np.linalg.norm(state-act[:2], axis=-1)
            action_node = np.argmin(dists)
            if start_position is not None:
                converted_actions.append((1,np.array([0.0,0.0]))) # mock release
            start_position = state[action_node]
            fully_observe.append(True)
        else:
            move = act[:2]-start_position
            converted_actions.append((action_node, move))
            start_position = act[:2]
            fully_observe.append(False)
    converted_actions.append((1,np.array([0.0,0.0])))
    fully_observe.append(True)
    return states[0], states, converted_actions, fully_observe

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='type of model to evaluate')
    parser.add_argument('--data', help='train, test, or real')
    parser.add_argument('--output', help='filename for saving figure and np arrays')
    parser.add_argument('--gin_config', help="path to gin config file.")
    parser.add_argument('--gin_bindings', action='append', help='gin bindings strings.')
    args = parser.parse_args()

    gin.parse_config_files_and_bindings([args.gin_config], args.gin_bindings)

    inferencer = HybridInference()

    if args.model == 'neural':
        simulator = neural_sim()
    elif args.model == 'physbam_2d':
        simulator = physbam_2d()
    elif args.model == 'physbam_3d':
        simulator = physbam_3d()

    # find sequences with length 100.
    train_index = []
    for i in range(9000):
        files = glob.glob('/scr1/mengyuan/data/sim3d_sequence/%04d/%04d_*.txt'%(i,i))
        if len(files)>100:
            train_index.append(i)
    test_index = []
    for i in range(9000, 10000):
        files = glob.glob('/scr1/mengyuan/data/sim3d_sequence/%04d/%04d_*.txt'%(i,i))
        if len(files)>100:
            test_index.append(i)
    print(len(train_index), len(test_index))

    start_time = time.time()

    batch_input_states, batch_actions, batch_gt_states = [], [], []

    if args.data == 'train':
        for i in train_index[:200]:
            input_state, states, actions = \
                load_sim_sequence('/scr1/mengyuan/data/sim3d_sequence/%04d'%(i), use_vision=False)
            batch_gt_states.append(states[:59])
            batch_actions.append(actions[:58])
            batch_input_states.append(input_state)
    elif args.data == 'test':
        for i in test_index[:200]:
            input_state, states, actions = \
                load_sim_sequence('/scr1/mengyuan/data/sim3d_sequence/%04d'%(i), use_vision=False)
            batch_gt_states.append(states[:59])
            batch_actions.append(actions[:58])
            batch_input_states.append(input_state)
    elif args.data == 'real':
        inferencer.memory = True
        for i in range(1,28):
            if os.path.isdir('/scr1/mengyuan/data/real_rope_with_occlusion-new/run_%d'%(i)):
                input_state, states, actions, fully_observe = \
                    load_real_sequence_new('/scr1/mengyuan/data/real_rope_with_occlusion-new/run_%d'%(i))
                batch_gt_states.append(states[:89])
                batch_actions.append(actions[:88])
                batch_input_states.append(input_state)

    print("load time: ", time.time()-start_time)
    start_time = time.time()

    state = []
    # to be tested for the new physbam_3d sim
    if args.model=='physbam_3d':
        idx = []
        for i,bis in enumerate(batch_input_states):
            try:
                st = state_to_mesh(bis)
                st = st.dot(np.array([[1,0,0],[0,0,1],[0,-1,0]]))
                state.append(st)
                idx.append(i)
            except NotImplementedError:
                print("sequence %d cannot construct mesh" %(i))
        batch_gt_states = [batch_gt_states[i] for i in idx]
        batch_actions = [batch_actions[i] for i in idx]
        batch_input_states = [batch_input_states[i] for i in idx]
    else:
        state = batch_input_states

    batch_gen_states = [batch_input_states]
    for i in range(len(batch_actions[0])):
        action = [[ac[i]] for ac in batch_actions]
        state = simulator.execute_batch(state, action, reset_spring=(i==0))
        if args.model=='physbam_3d':
            obs = [0.5*(st[:64]+st[64:]) for st in state]
            if batch_input_states[0].shape[-1]==2:
                obs = [ob[:,:2] for ob in obs]
            batch_gen_states.append(obs)
        else:
            batch_gen_states.append(state)
    batch_gen_states = np.array(batch_gen_states).transpose((1,0,2,3))
    print("prediction time: ", time.time()-start_time)
    start_time = time.time()

    batch_gt_states=np.array(batch_gt_states)
    dists = np.linalg.norm(batch_gen_states-batch_gt_states, axis=-1)
    avg_dists = np.mean(dists, axis=-1)
    max_dists = np.amax(dists, axis=-1)
    print("eval time: ", time.time()-start_time)

    avg_dists_mean = np.mean(avg_dists, axis=0)
    avg_dists_std = np.std(avg_dists, axis=0)
    max_dists_mean = np.mean(max_dists, axis=0)
    max_dists_std = np.std(max_dists, axis=0)

    plt.figure() # without offseting the start dists.
    plt.plot(avg_dists_mean, c='C0')
    plt.fill_between(np.arange(batch_gen_states.shape[1]),
                     avg_dists_mean-avg_dists_std,
                     avg_dists_mean+avg_dists_std,
                     alpha = 0.3, facecolor='C0')
    plt.plot(max_dists_mean, c='C1')
    plt.fill_between(np.arange(batch_gen_states.shape[1]),
                     max_dists_mean-max_dists_std,
                     max_dists_mean+max_dists_std,
                     alpha = 0.3, facecolor='C1')
    plt.axis([0,batch_gen_states.shape[1],0,0.5])
    plt.savefig(args.output+'.png') #('eval_sim_trainset_withvision.png')
    np.savez(args.output+'.npz', #('eval_sim_trainset_withvision.npz',
             mean_avg=avg_dists_mean,
             std_avg=avg_dists_std,
             mean_max=max_dists_mean,
             std_max=max_dists_std)

