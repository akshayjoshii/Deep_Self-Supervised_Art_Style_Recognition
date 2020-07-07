import logging
import torch
import torch.nn as nn
import torchvision
from torchvision import models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from collections import Counter

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


#input_size = 32 * 32 * 3
layer_config= [512, 256]
num_classes = 27
num_epochs = 20
batch_size = 100
learning_rate = 1e-3
learning_rate_decay = 0.9
fine_tune = True
pretrained=True
# num_training= 35 #45000
# num_validation = 22 #15000
# num_test = 5 #20000
# lengths = [num_training, num_validation, num_test] 
reg=0.001

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s'%device)


data_aug_transforms = transforms.Compose([transforms.RandomCrop(500, pad_if_needed=True), transforms.ToTensor()])

def load_dataset():
    data_path = '/src/wikiart'
    #data_path = '/content/wikitest'
    dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=data_aug_transforms
    )
    #train_mask = list(range(num_training))
    length = len(dataset)
    num_training= int(0.6 * length)
    num_validation = int(0.2 * length)
    num_test = length - num_training - num_validation
    lengths = [num_training, num_validation, num_test] 

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, lengths)
    #print(len(train_dataset))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True
    )
    #val_mask = list(range(num_training, num_training + num_validation))
    #val_dataset = torch.utils.data.Subset(dataset, mask)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True
    )
    return train_loader, val_loader, test_loader, test_dataset

"""
def load_testset():
    #data_path = '/src/wikiart'
    
    # test_dataset = torchvision.datasets.ImageFolder(
    #     root=data_path,
    #     transform=data_aug_transforms
   # )
    
    return test_dataset,test_loader
"""

train_loader, val_loader, test_loader, test_dataset = load_dataset()
#test_dataset,test_loader = load_testset()

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

class ResNetModel(nn.Module):
    def __init__(self, n_class, fine_tune, pretrained=True):
        super(ResNetModel, self).__init__()
        
        resnet = models.resnet101(pretrained)

        set_parameter_requires_grad(resnet, fine_tune)
        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Linear(num_ftrs, num_classes)
        self.model = resnet
    

    def forward(self, x):
       
        out = self.model(x)
      
        return out

# Initialize the model for this run
model= ResNetModel(num_classes, fine_tune, pretrained)
#print(model)
model.to(device)
params_to_update = model.parameters()
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params_to_update, lr=learning_rate, weight_decay=reg)
#train
best_accuracy = 0
ResN_Val_Acc = []
ResN_Losses = []
lr = learning_rate
total_step = len(train_loader)
logging.basicConfig(filename='/src/Model_Training.log', level=logging.DEBUG)
for epoch in range(num_epochs):
  #for i, (images, labels) in enumerate(train_loader):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # if i==0:
        #     imageiter = iter(images)
        #     image = next(imageiter)
        #     print(image.shape)
        #     plt.imshow(image.cpu().permute(1, 2, 0))
        #     print(labels)

        outputs = model(images)
        loss = criterion(outputs, labels)
        ResN_Losses.append(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 50 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            logging.debug('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}\n'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    
    lr *= learning_rate_decay
    update_lr(optimizer, lr)

    #Added model.eval() for Validation
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            #Val = 100 * (correct / total)
        print('Validataion accuracy for {} images is: {} %'.format(total, 100 * (correct / total)))
        logging.debug('Validataion accuracy for {} images is: {} %\n'.format(total, 100 * (correct / total)))
        
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    predicted_all=[]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        predicted_all.append(predicted)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if total == 1000:
          break
    print('Accuracy of the final network on {} test images: {} %'.format(total, 100 * (correct / total)))
    logging.debug('Accuracy of the final network on {} test images: {} %\n'.format(total, 100 * (correct / total)))
