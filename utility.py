import config as cfg
import pickle
import os
import matplotlib.pyplot as plt


file = os.path.join(cfg.metric_directory, 'siam_mlp', 'embeddings_ones', 'epoch', cfg.pickle_file)
with open(file, 'rb') as f:
    model_coops = pickle.load(f)['coops_dict']['siam_mlp']

fig, ax = plt.subplots(1, 3)
plt.suptitle(r'embeddings ~{1}$^{64}$')
for i, (key, value) in enumerate(model_coops.items()):
    ax[i].set_title(key)
    ax[i].hist(value, bins=5)
    ax[i].set_ylim(0., 800.)

plt.show()
