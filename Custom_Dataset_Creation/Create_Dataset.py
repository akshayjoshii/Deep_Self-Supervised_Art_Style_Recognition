import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import pickle
import random
# from torchvision.transforms.functional import resize
# import torchvision.transforms as transforms


# Datapath = "..\wikiart"
Datapath = "/src/wikiart"
Categories = ["Abstract_Expressionism", "Action_painting", "Analytical_Cubism", "Art_Nouveau_Modern",
                "Baroque", "Color_Field_Painting", "Contemporary_Realism", "Cubism", "Early_Renaissance", "Expressionism",
                "Fauvism", "High_Renaissance", "Impressionism", "Mannerism_Late_Renaissance", "Minimalism", "Naive_Art_Primitivism",
                "New_Realism", "Northern_Renaissance", "Pointillism", "Pop_Art", "Post_Impressionism", "Realism", "Rococo", "Romanticism",
                "Symbolism", "Synthetic_Cubism", "Ukiyo_e"]

# Categories = ["Abstract_Expressionism", "Action_painting", "Analytical_Cubism", "Art_Nouveau_Modern",
#                 "Baroque", "Color_Field_Painting", "Contemporary_Realism", "Cubism", "Early_Renaissance", "Expressionism",
#                 "Fauvism", "High_Renaissance", "Impressionism", "Mannerism_Late_Renaissance", "Minimalism"]

# Categories = ['Abstract_Expressionism', 'Action_painting', 'Cubism', 'Minimalism', 'Expressionism', 'Analytical_Cubism']
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
test_data = []
dict_train = {'labels': [], 'data': []}
dict_test = {'labels': [], 'data': []}

def create_dataset():
    total_count = 0
    total_train = 0
    total_test = 0
    cls = 1

    for category in Categories:
        path = os.path.join(Datapath, category)
        class_num = Categories.index(category)
        print('class:', cls, category)
        cls += 1
        c = 0
        list_of_files_present = os.listdir(path)

        total_images_present = len(list_of_files_present)
        max_images_count = 1400
        if total_images_present >= max_images_count:
            images_to_take = random.sample(list_of_files_present, max_images_count)
            train_size = 1000
            test_size = 400
        else:
            images_to_take = random.sample(list_of_files_present, total_images_present)
            test_size = int(0.2857 * total_images_present)
            train_size = total_images_present - test_size

        print('total_images_present:', total_images_present)
        print('total images taken:', max_images_count if max_images_count < total_images_present else total_images_present)
        print('train_size', train_size)
        total_train += train_size
        print('test_size', test_size)
        total_test += test_size

        for img in tqdm(images_to_take):
            if c > max_images_count or c > total_images_present:
                break
            if c < train_size:
                try:
                    img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_COLOR)

                    #for scaling with aspect ratio. (downscaling 60%)
                    # scale_percent = 60  # percent of original size
                    # width = int(img_array.shape[1] * scale_percent / 100)
                    # height = int(img_array.shape[0] * scale_percent / 100)
                    # dim = (height, width)
                    # print(dim)
                    # resize image
                    new_array = cv2.resize(img_array, (384, 384))
                    # print(len(new_array))
                    # print(len(new_array.flatten()))
                    # print(new_array.flatten())
                    # break
                    dict_train['labels'].append(class_num)
                    dict_train['data'].append(new_array.flatten())
                except Exception as e:
                    pass
            else:
                try:
                    img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_COLOR)

                    # scale_percent = 60  # percent of original size
                    # width = int(img_array.shape[1] * scale_percent / 100)
                    # height = int(img_array.shape[0] * scale_percent / 100)
                    # dim = (height, width)
                    # resize image
                    new_array = cv2.resize(img_array, (384, 384))
                    dict_test['labels'].append(class_num)
                    dict_test['data'].append(new_array.flatten())
                except Exception as e:
                    pass

            c += 1
        total_count += c
        print('no. of images processed currently: ', total_count)
    print('Total training images taken:', total_train)
    print('Total test images taken:', total_test)
    return dict_train, dict_test

#Driver Code
# final_train, final_test = [], []
final_train, final_test = create_dataset()

# Shuffling the data
for i in range(0, 2):
    temp = list(zip(dict_train['data'], dict_train['labels']))
    random.shuffle(temp)
    res1, res2 = zip(*temp)
    dict_train['data'] = res1
    dict_train['labels'] = res2

    temp = list(zip(dict_test['data'], dict_test['labels']))
    random.shuffle(temp)
    res1, res2 = zip(*temp)
    dict_test['data'] = res1
    dict_test['labels'] = res2

# print(len(dict_train['data']))
# print(len(dict_train['labels']))
print('length of final train dict data value', len(dict_train['data'][0]))
print('length of final test dict data value', len(dict_test['data'][0]))
print('pickle is generated inside ')
# print(dict_train['data'])
# print(dict_train['labels'])

#Dump a pickle dataset file
# pickle_out = open("..\wikiart\\train_1000.pickle","wb")
pickle_out = open("/src/wikiart\\train_1000.pickle", "wb")
pickle.dump(dict_train, pickle_out)
pickle_out.close()

# pickle_out = open("..\wikiart\\test_1000.pickle","wb")
pickle_out = open("/src/wikiart\\test_1000.pickle", "wb")
pickle.dump(dict_test, pickle_out)
pickle_out.close()

# Read the dumped dataset
# pickle_in = open("..\wikiart\\train_data.pickle","rb")
# pickled_dataset = pickle.load(pickle_in)
# print ("Dumping complete")
# for sample in pickled_dataset[:5]:
#     print(sample[0])


''' dump code for transformations
# train_transforms = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize(img_array, (400, 400)),
#     transforms.ToTensor()])
# new_array = train_transforms(img_array)
# n = transforms.Resize(img_array, (400, 400))
# nn = transforms.ToPILImage(n)
# new_array = transforms.ToTensor(nn)
# new_array = cv2.resize(img_array, (400, 400))
train_data.append([new_array, class_num])


#Driver Code
# final_train, final_test = [], []
# final_train, final_test = create_dataset()

# #Length of the numpy array
# print(len(final_train))
# print(len(final_test))
# 
# #Random shuffle  the images for better training
# random.shuffle(final_train)
# random.shuffle(final_test)
# 
# #Check if we have a good mix of classes
# for sample in final_train[:5]:
#     print(sample[1])
# 
# for sample in final_test[:5]:
#     print(sample[1])
'''