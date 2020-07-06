import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import pickle
import random

#Comment
pickled_dataset = []
pickle_in = open("F:\wikiart\cifar-10-batches-py\data_batch_1","rb")
pickled_dataset = pickle.load(pickle_in, encoding='latin1')
#print ("Dumping complete")
features = pickled_dataset['data'].reshape((len(pickled_dataset['data']), 3, 32, 32))
name = pickled_dataset['filenames']

labels = pickled_dataset['labels']
print(pickled_dataset)
"""
for label, filenames in pickled_dataset:
    plt.imshow(filenames)
    plt.show()
    #print(sample[1])

"""