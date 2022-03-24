#!/usr/bin/env python
# coding: utf-8

# In[]:


import numpy as np
import os
import tensorflow as tf

from collections import Counter
from pathlib import Path
from split_data_utils import get_split_results, output_to_files

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

task = 'cifar10'
splita_name = "non_iid_4"
distrib = non_iid_4

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
output_to_files(train_split_results, task=task, splita=splita_name, suffix="train")
output_to_files(test_split_results, task=task, splita=splita_name, suffix="test")
