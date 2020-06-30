import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import pickle
import random

#Comment one 2 3
Datapath = "/src/wikiart"
Categories = ["Abstract_Expressionism", "Action_painting", "Analytical_Cubism", "Art_Nouveau_Modern", 
                "Baroque", "Color_Field_Paigit nting", "Contemporary_Realism", "Cubism", "Early_Renaissance", "Expressionism",
                "Fauvism", "High_Renaissance", "Impressionism", "Mannerism_Late_Renaissance", "Minimalism", "Naive_Art_Primitivism", 
                "New_Realism", "Northern_Renaissance", "Pointillism", "Pop_Art", "Post_Impressionism", "Realism", "Rococo", "Romanticism",
                "Symbolism", "Synthetic_Cubism", "Ukiyo_e"]

"""
##Code to visualize each image in the wikiart dataset folder##

for category in Categories:
    path = os.path.join(Datapath, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
        new_array = cv2.resize(img_array, (500, 500))
        plt.imshow(new_array)
        plt.show()
        break
    break
"""

#Function to create custom Dataset
train_data = []
def create_dataset():
    for category in Categories:
        path = os.path.join(Datapath, category)
        class_num = Categories.index(category)
        n = 0
        for img in tqdm(os.listdir(path)): 
            try:
                if n < 500:
                    n += 1
                    img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_COLOR) 
                    new_array = cv2.resize(img_array, (500, 500))  
                    train_data.append([new_array, class_num])
                else:
                    pass
            except Exception as e:
                pass
    return train_data


#Driver Code
final = []
final = create_dataset()

#Length of the numpy array
print(len(final))

#Random shuffle  the images for better training
random.shuffle(final)

#Check if we have a good mix of classes
for sample in final[:5]:
    print(sample[1])

#Dump a pickle dataset file
pickle_out = open("/src/Dataset.pickle","wb")
pickle.dump(final, pickle_out)
pickle_out.close()

#Read the dumped dataset
pickle_in = open("/src/Dataset.pickle","rb")
pickled_dataset = pickle.load(pickle_in)
print ("Dumping complete")
#for sample in pickled_dataset[:5]:
    #print(sample[0])
