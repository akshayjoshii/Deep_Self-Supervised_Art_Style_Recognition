import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import pickle
import random

#Comment
pickled_dataset = []
pickle_in = open("/src/Dataset.pickle","rb")
pickled_dataset = pickle.load(pickle_in)
#print ("Dumping complete")
for sample in pickled_dataset[5:]:
    plt.imshow(sample[0])
    plt.show()
    print(sample[1])
