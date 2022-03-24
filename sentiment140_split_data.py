#!/usr/bin/env python
# coding: utf-8

# In[]:


import numpy as np
import scipy.sparse as sp
from split_data_utils import get_split_results, output_to_files

# In[]:

iid_4 = {
    0: [1, 1, 1, 1],
    4: [1, 1, 1, 1],
}

non_iid_4 = {
    0: [4, 4, 1, 1],
    4: [1, 1, 4, 4],
}

task_name = "sentiment140"
splita_name = "non_iid_4"
distrib = non_iid_4

# In[]:

x_train = sp.load_npz('./data/sentiment140_x_train.csr.npz').toarray()
y_train = np.load('./data/sentiment140_y_train.npy')

x_test = sp.load_npz('./data/sentiment140_x_test.csr.npz').toarray()
y_test = np.load('./data/sentiment140_y_test.npy')

# In[]:


train_split_results = get_split_results(distribution=distrib, x=x_train, y=y_train, verbose=True)
test_split_results = get_split_results(distribution=distrib, x=x_test, y=y_test, verbose=True)
output_to_files(train_split_results, task='sentiment140', splita=splita_name, suffix="train")
output_to_files(test_split_results, task='sentiment140', splita=splita_name, suffix="test")

# %%
