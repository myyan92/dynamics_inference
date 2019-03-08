""" Starting from a image, test fitting_inference/model_inference gives
reasonable output, correctly converted to robot frame.
Both estimated state and action in robot frame are given to simulation.
Test conversion to simulator coordinate is correct, simulation is reasonable,
and conversion back to robot frame is correct.
"""

import numpy as np
import matplotlib.pyplot as plt
from physbam_python_new import rollout_physbam
from sim_integration.fitting_inference import FittingInference
import os, pdb
from PIL import Image

test_data_dir = "/scr1/mengyuan/data/real_rope_ours_2/seq_m2_2/"

actions = np.load(os.path.join(test_data_dir, 'actions.npy'))
actions = actions[:,[12,13,16,17]] # starting position and move delta
images = []
num_images = actions.shape[0]+1
for i in range(num_images):
    file = os.path.join(test_data_dir, 'image_%d.png'%(i))
    im = Image.open(file)
    images.append(np.array(im))

inferencer = FittingInference()
inferencer.set_guess(images[0])

for before, after, action in zip(images[0:-1], images[1:], actions):
    state = inferencer.inference(before)
    dists = np.linalg.norm(action[:2]-state, axis=-1)
    action_node = np.argmin(dists)
    dist = np.amin(dists)
    if dist > 0.03:
        print("action not in contact. Skip simulation")
        next_state = state
    else:
        trans_action = np.array([[action[2], action[3], float(action_node)/state.shape[0]]])
        print(state)
        print(trans_action)
        next_state = rollout_physbam.rollout_single(state, trans_action,
                physbam_args=" -friction 0.176 -stiffen_linear 2.223 -stiffen_bending 0.218")
    # To image frame and visualize
    next_state_im = next_state.copy()
    next_state_im[:,1] *= -1.0
    next_state_im[:,1] += 0.5
    next_state_im *= after.shape[0]
    plt.imshow(after)
    plt.plot(next_state_im[:,0], next_state_im[:,1])
    plt.show()
