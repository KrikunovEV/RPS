import config as cfg
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np


n3_equ_steps_distr = os.path.join(cfg.metric_directory, 'siam_mlp', 'distributions_diff_steps', 'epoch',
                                  cfg.pickle_file)
with open(n3_equ_steps_distr, 'rb') as f:
    model_coops = pickle.load(f)['coops_dict']['siam_mlp']

fig, ax = plt.subplots(1, 3)
plt.suptitle('2 & 3 agents have more negotiation steps')
for i, (key, value) in enumerate(model_coops.items()):
    ax[i].set_title(key)
    ax[i].hist(value, bins=10)
    ax[i].set_ylim(0., 125.)

plt.show()
