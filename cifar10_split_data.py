#!/usr/bin/env python
# coding: utf-8

# In[]:


import numpy as np
import os
import tensorflow as tf

from collections import Counter
from pathlib import Path

# In[]:


iid_4 = {
    0: [1, 1, 1, 1],
    1: [1, 1, 1, 1],
    2: [1, 1, 1, 1],
    3: [1, 1, 1, 1],
    4: [1, 1, 1, 1],
    5: [1, 1, 1, 1],
    6: [1, 1, 1, 1],
    7: [1, 1, 1, 1],
    8: [1, 1, 1, 1],
    9: [1, 1, 1, 1],
}

non_iid_4 = {
    0: [7, 1, 1, 1],
    1: [1, 7, 1, 1],
    2: [1, 1, 7, 1],
    3: [1, 1, 1, 7],

    4: [2, 1, 1, 1],
    5: [1, 2, 1, 1],
    6: [1, 1, 2, 1],
    7: [1, 1, 1, 2],

    8: [1, 1, 1, 1],
    9: [1, 1, 1, 1],
}

splita_name = "iid_4"
distrib = iid_4


# In[]:


def get_split_results(distribution, x, y, verbose=False):
    label_all_images = [x[y == idx] for idx in range(10)]
    if verbose:
        print('Label distribution:', [len(i) for i in label_all_images])
    split_results = [[], [], [], []]
    for label, splits in distribution.items():
        a = np.cumsum(splits, dtype=np.float64)
        a = (a / a[-1] * len(label_all_images[label]))[:-1]
        a = np.round(a).astype(np.int64)
        res = np.split(label_all_images[label], a)
        for i, r in enumerate(res):
            for img in r:
                split_results[i].append((img, label))
    if verbose:
        for i, res in enumerate(split_results):
            counter = Counter(i for _, i in res)
            print(f'  node-{i} got {len(res)} samples', sorted(counter.items()))
    return split_results


# In[]:


def output_to_files(split_results, splita: str, suffix: str = ""):
    output_dir = Path(f'./data/splits/{splita}/')
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, split_result in enumerate(split_results):
        node_x, node_y = zip(*split_result)
        node_x = np.array(node_x, dtype=np.uint8)
        node_y = np.array(node_y, dtype=np.uint8)
        np.savez_compressed(output_dir.joinpath(f'cifar10_node-{i}_x_{suffix}.npz'), node_x)
        np.savez_compressed(output_dir.joinpath(f'cifar10_node-{i}_y_{suffix}.npz'), node_y)


# In[]:


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train = np.squeeze(y_train, axis=1)
y_test = np.squeeze(y_test, axis=1)

# In[]:


v6_x_test = np.load('./data/cifar10.1_v6_data.npy')
v6_y_test = np.load('./data/cifar10.1_v6_labels.npy')
print(v6_x_test.shape)
print(v6_y_test.shape)

# In[]:


x_test = np.concatenate([x_test, v6_x_test], axis=0)
y_test = np.concatenate([y_test, v6_y_test], axis=0)
print(x_test.shape)

# In[]:


train_split_results = get_split_results(distribution=distrib, x=x_train, y=y_train, verbose=True)
test_split_results = get_split_results(distribution=distrib, x=x_test, y=y_test, verbose=True)
output_to_files(train_split_results, splita=splita_name, suffix="train")
output_to_files(test_split_results, splita=splita_name, suffix="test")
