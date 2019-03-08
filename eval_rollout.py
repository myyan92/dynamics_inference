import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os, pdb, time, glob
from PIL import Image
import argparse, gin
from visual_inference.hybrid_inference import HybridInference
from TF_cloth2d.models.model_VGG_STN_2 import Model_STNv2
from dynamics_inference.dynamic_models import *


inferencer = HybridInference()

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
        im = Image.open(os.path.join(dir, index+'_000.png'))
        image = np.array(im)[::-1,:,:] # for seq3d data
        inferencer.fitting.init_position=None # reset memory to be safe
        input_state = inferencer.inference(image)
        # convert frame
        # align to true state
    else:
        input_state = states[0]
    return input_state, states[1:], actions[1:]

def load_real_sequence(dir):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='type of model to evaluate')
    parser.add_argument('--data', help='train, test, or real')
    parser.add_argument('--output', help='filename for saving figure and np arrays')
    parser.add_argument('--gin_config', help="path to gin config file.")
    parser.add_argument('--gin_bindings', action='append', help='gin bindings strings.')
    args = parser.parse_args()

    gin.parse_config_files_and_bindings([args.gin_config], args.gin_bindings)

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
                load_sim_sequence('/scr1/mengyuan/data/sim3d_sequence/%04d'%(i), use_vision=True)
            batch_gt_states.append(states[:59])
            batch_actions.append(actions[:58])
            batch_input_states.append(input_state)
    elif args.data == 'test':
        for i in test_index[:200]:
            input_state, states, actions = \
                load_sim_sequence('/scr1/mengyuan/data/sim3d_sequence/%04d'%(i), use_vision=True)
            batch_gt_states.append(states[:59])
            batch_actions.append(actions[:58])
            batch_input_states.append(input_state)
    elif args.data == 'real':
        inferencer.memory = True
        for i in [1,2,3,5,8,9]:
            input_state, states, actions = \
                load_real_sequence('/scr1/mengyuan/data/real_rope_ours_2/seq_m%d_2'%(i))
            batch_gt_states.append(states[:59])
            batch_actions.append(actions[:58])
            batch_input_states.append(input_state)

    print("load time: ", time.time()-start_time)
    start_time = time.time()

    batch_gen_states = [batch_input_states]
    state = batch_input_states
    for i in range(58):
        action = [[ac[i]] for ac in batch_actions]
        state = simulator.execute_batch(state, action)
        batch_gen_states.append(state)
    batch_gen_states = np.array(batch_gen_states).transpose((1,0,2,3))
    print("prediction time: ", time.time()-start_time)
    start_time = time.time()

    avg_dists, max_dists = [], []
    for gen_states, states in zip(batch_gen_states, batch_gt_states):
        avg_d, max_d = [], []
        for pred, gt in zip(gen_states, states):
            dists = np.linalg.norm(pred-gt, axis=-1)
            avg_d.append(np.mean(dists))
            max_d.append(np.amax(dists))
        avg_dists.append(avg_d)
        max_dists.append(max_d)
    print("eval time: ", time.time()-start_time)

    plt.figure() # without offseting the start dists.
    plt.plot(np.mean(avg_dists, axis=0), c='C0')
    plt.fill_between(np.arange(59),
                     np.mean(avg_dists, axis=0)-np.std(avg_dists, axis=0),
                     np.mean(avg_dists, axis=0)+np.std(avg_dists, axis=0),
                     alpha = 0.3, facecolor='C0')
    plt.plot(np.mean(max_dists, axis=0), c='C1')
    plt.fill_between(np.arange(59),
                     np.mean(max_dists, axis=0)-np.std(max_dists, axis=0),
                     np.mean(max_dists, axis=0)+np.std(max_dists, axis=0),
                     alpha = 0.3, facecolor='C1')
    plt.axis([0,60,0,0.5])
    plt.savefig(args.output+'.png') #('eval_sim_trainset_withvision.png')
    np.savez(args.output+'.npz', #('eval_sim_trainset_withvision.npz',
             mean_avg=np.mean(avg_dists, axis=0),
             std_avg=np.std(avg_dists, axis=0),
             mean_max=np.mean(max_dists, axis=0),
             std_max=np.std(max_dists, axis=0))

