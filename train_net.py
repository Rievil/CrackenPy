# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 13:32:27 2023
from: https://www.kaggle.com/code/ligtfeather/semantic-segmentation-is-easy-with-pytorch/notebook
@author: Richard
"""


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable

from PIL import Image
import cv2
import albumentations as A

import time
import os
from tqdm.notebook import tqdm

# !pip install -q segmentation-models-pytorch
# !pip install -q torchsummary

from torchsummary import summary
import segmentation_models_pytorch as smp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# IMAGE_PATH = r'C:\Users\Richard\Vysoké učení technické v Brně\22-02098S - Dokumenty\General\Data\Kamera\DataSets\GACR\GACR_CrackDataset\Images'
# MASK_PATH = r'C:\Users\Richard\Vysoké učení technické v Brně\22-02098S - Dokumenty\General\Data\Kamera\DataSets\GACR\GACR_CrackDataset\Labels'

IMAGE_PATH = r'C:\PyTorchData\GACR_CrackDataset\Images'
MASK_PATH = r'C:\PyTorchData\GACR_CrackDataset\Labels'

#%%

# n_classes = 22

n_classes = 5 

def create_df():
    name = []
    for dirname, _, filenames in os.walk(IMAGE_PATH):
        for filename in filenames:
            name.append(filename.split('.')[0])
    
    return pd.DataFrame({'id': name}, index = np.arange(0, len(name)))

df = create_df()
print('Total Images: ', len(df))

#% split data

X_trainval, X_test = train_test_split(df['id'].values, test_size=0.1, random_state=19)
X_train, X_val = train_test_split(X_trainval, test_size=0.15, random_state=19)

print('Train Size   : ', len(X_train))
print('Val Size     : ', len(X_val))
print('Test Size    : ', len(X_test))

#% Show image
ids=101
img = Image.open(IMAGE_PATH + '\\' + df['id'][ids] + '.png')
mask = Image.open(MASK_PATH + '\\' + df['id'][ids] + '.png')
print('Image Size', np.asarray(img).shape)
print('Mask Size', np.asarray(mask).shape)


plt.imshow(img)
plt.imshow(mask, alpha=0.6)
plt.title('Picture with Mask Appplied')
plt.show()

#%%

class DroneDataset(Dataset):
    
    def __init__(self, img_path, mask_path, X, mean, std, transform=None, patch=False):
        self.img_path = img_path
        self.mask_path = mask_path
        self.X = X
        self.transform = transform
        self.patches = patch
        self.mean = mean
        self.std = std
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.img_path + '\\'  + self.X[idx] + '.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_path + '\\'  + self.X[idx] + '.png', cv2.IMREAD_GRAYSCALE)
        
        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = Image.fromarray(aug['image'])
            mask = aug['mask']
        
        if self.transform is None:
            img = Image.fromarray(img)
        
        t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
        img = t(img)
        mask = torch.from_numpy(mask).long()
        
        if self.patches:
            img, mask = self.tiles(img, mask)
            
        return img, mask
    
    def tiles(self, img, mask):

        img_patches = img.unfold(1, 512, 512).unfold(2, 768, 768) 
        img_patches  = img_patches.contiguous().view(3,-1, 512, 768) 
        img_patches = img_patches.permute(1,0,2,3)
        
        mask_patches = mask.unfold(0, 512, 512).unfold(1, 768, 768)
        mask_patches = mask_patches.contiguous().view(-1, 512, 768)
        
        return img_patches, mask_patches
    
#%%
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

t_train = A.Compose([A.Resize(416, 416, interpolation=cv2.INTER_NEAREST), A.HorizontalFlip(), A.VerticalFlip(), 
                     A.GridDistortion(p=0.2), A.RandomBrightnessContrast((0,0.5),(0,0.5)),
                     A.GaussNoise()])

t_val = A.Compose([A.Resize(416, 416, interpolation=cv2.INTER_NEAREST), A.HorizontalFlip(),
                   A.GridDistortion(p=0.2)])

#datasets
train_set = DroneDataset(IMAGE_PATH, MASK_PATH, X_train, mean, std, t_train, patch=False)
val_set = DroneDataset(IMAGE_PATH, MASK_PATH, X_val, mean, std, t_val, patch=False)


#dataloader
batch_size= 6 

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)     

#%%

# model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True)
# model = smp.FPN('resnet101', in_channels=3,classes=5,activation=None, encoder_depth=5)
model = smp.FPN('resnext101_32x8d', in_channels=3,classes=5,activation=None, encoder_depth=5)
# model = smp.FPN('timm-regnety_160', in_channels=3,classes=5,activation=None, encoder_depth=5)

# model = smp.Unet('mobilenet_v2', encoder_weights='imagenet', classes=n_classes, activation=None, encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16])

#%%

def pixel_accuracy(output, mask):
    with torch.no_grad():
        output = torch.argmax(F.softmax(output, dim=1), dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy

def mIoU(pred_mask, mask, smooth=1e-10, n_classes=4):
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(0, n_classes): #loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0: #no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union +smooth)
                iou_per_class.append(iou)
        return np.nanmean(iou_per_class)
    

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit(epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, patch=False):
    torch.cuda.empty_cache()
    train_losses = []
    test_losses = []
    val_iou = []; val_acc = []
    train_iou = []; train_acc = []
    lrs = []
    min_loss = np.inf
    decrease = 1 ; not_improve=0

    model.to(device)
    fit_time = time.time()
    for e in range(epochs):
        since = time.time()
        running_loss = 0
        iou_score = 0
        accuracy = 0
        #training loop
        model.train()
        
        for i, data in enumerate(tqdm(train_loader)):
            #training phase
            image_tiles, mask_tiles = data
            if patch:
                bs, n_tiles, c, h, w = image_tiles.size()

                image_tiles = image_tiles.view(-1,c, h, w)
                mask_tiles = mask_tiles.view(-1, h, w)
            
            image = image_tiles.to(device); mask = mask_tiles.to(device);
            #forward
            output = model(image)
            loss = criterion(output, mask)
            #evaluation metrics
            iou_score += mIoU(output, mask)
            accuracy += pixel_accuracy(output, mask)
            #backward
            loss.backward()
            optimizer.step() #update weight          
            optimizer.zero_grad() #reset gradient
            
            #step the learning rate
            lrs.append(get_lr(optimizer))
            scheduler.step() 
            
            running_loss += loss.item()
            
        else:
            model.eval()
            test_loss = 0
            test_accuracy = 0
            val_iou_score = 0
            #validation loop
            with torch.no_grad():
                for i, data in enumerate(tqdm(val_loader)):
                    #reshape to 9 patches from single image, delete batch size
                    image_tiles, mask_tiles = data

                    if patch:
                        bs, n_tiles, c, h, w = image_tiles.size()

                        image_tiles = image_tiles.view(-1,c, h, w)
                        mask_tiles = mask_tiles.view(-1, h, w)
                    
                    image = image_tiles.to(device); mask = mask_tiles.to(device);
                    output = model(image)
                    #evaluation metrics
                    val_iou_score +=  mIoU(output, mask)
                    test_accuracy += pixel_accuracy(output, mask)
                    #loss
                    loss = criterion(output, mask)                                  
                    test_loss += loss.item()
            
            #calculatio mean for each batch
            train_losses.append(running_loss/len(train_loader))
            test_losses.append(test_loss/len(val_loader))


            if min_loss > (test_loss/len(val_loader)):
                print('Loss Decreasing.. {:.3f} >> {:.3f} '.format(min_loss, (test_loss/len(val_loader))))
                min_loss = (test_loss/len(val_loader))
                decrease += 1
                if decrease % 5 == 0:
                    print('saving model...')
                    torch.save(model, 'Unet-Mobilenet_v2_mIoU-{:.3f}.pt'.format(val_iou_score/len(val_loader)))
                    

            if (test_loss/len(val_loader)) > min_loss:
                not_improve += 1
                min_loss = (test_loss/len(val_loader))
                print(f'Loss Not Decrease for {not_improve} time')
                if not_improve == 7:
                    print('Loss not decrease for 7 times, Stop Training')
                    break
            
            #iou
            val_iou.append(val_iou_score/len(val_loader))
            train_iou.append(iou_score/len(train_loader))
            train_acc.append(accuracy/len(train_loader))
            val_acc.append(test_accuracy/ len(val_loader))
            print("Epoch:{}/{}..".format(e+1, epochs),
                  "Train Loss: {:.3f}..".format(running_loss/len(train_loader)),
                  "Val Loss: {:.3f}..".format(test_loss/len(val_loader)),
                  "Train mIoU:{:.3f}..".format(iou_score/len(train_loader)),
                  "Val mIoU: {:.3f}..".format(val_iou_score/len(val_loader)),
                  "Train Acc:{:.3f}..".format(accuracy/len(train_loader)),
                  "Val Acc:{:.3f}..".format(test_accuracy/len(val_loader)),
                  "Time: {:.2f}m".format((time.time()-since)/60))
        
    history = {'train_loss' : train_losses, 'val_loss': test_losses,
               'train_miou' :train_iou, 'val_miou':val_iou,
               'train_acc' :train_acc, 'val_acc':val_acc,
               'lrs': lrs}
    print('Total time: {:.2f} m' .format((time.time()- fit_time)/60))
    return history
#%%
torch.cuda.empty_cache()
#%% Train

max_lr = 1e-3
epoch = 15
weight_decay = 1e-4

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epoch,
                                            steps_per_epoch=len(train_loader))

history = fit(epoch, model, train_loader, val_loader, criterion, optimizer, sched)

#%%

torch.save(model.state_dict(), r'Models\resnext101_32x8d_N387_C5_30102023.pt')
#%%

# model.load_state_dict(torch.load('Unet-Mobilenet_Cracks.pt'))

model.load_state_dict(torch.load(r'Models\resnext101_32x8d_N387_C5.pt'))
model.eval()


#%% 
fig,(ax1,ax2)=plt.subplots(1,2)
#%

# img_path=r'C:\PyTorchData\GACR_CrackDataset\Images\ID1_Spec1200_107_Image.png'
img_path=r'C:\PyTorchData\ConcreteCracksMat\Img\Label_122.png'
# img_path=r'C:\PyTorchData\Test\Img5.png'

def GetImg(impath):
    img = cv2.imread(impath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(416, 416), interpolation=cv2.INTER_NEAREST)
    img = Image.fromarray(img)
    return img

def predict_image(model, image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    model.eval()
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    image = t(image)
    model.to(device); image=image.to(device)
    with torch.no_grad():
        image = image.unsqueeze(0)
        output = model(image)

        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked

img=GetImg(img_path)
ax1.imshow(img)
mask=predict_image(model,img)
ax2.imshow(mask)
plt.savefig("Plots\Example of classification.png")
#%%

model_weitghs=str("")
#%%

model2 = smp.Unet('mobilenet_v2', encoder_weights='imagenet', classes=n_classes, activation=None, encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16])
model_weitghs = torch.load('Unet-Mobilenet_Cracks.pt')

model2.load_state_dict(model_weitghs)
model2.eval()

#%%
import pandas as pd

def plot_loss(history):
    plt.plot(history['val_loss'], label='val', marker='o')
    plt.plot( history['train_loss'], label='train', marker='o')
    
    
    plt.title('Loss per epoch'); plt.ylabel('loss');
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.show()
    
# def plot_score(history):
    plt.plot(history['train_miou'], label='train_mIoU', marker='*')
    plt.plot(history['val_miou'], label='val_mIoU',  marker='*')
    plt.title('Score per epoch'); plt.ylabel('mean IoU')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.show()
    
# def plot_acc(history):
    plt.plot(history['train_acc'], label='train_accuracy', marker='*')
    plt.plot(history['val_acc'], label='val_accuracy',  marker='*')
    plt.title('Accuracy per epoch'); plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.show()
    
    df=pd.DataFrame({"val_loss":history['val_loss'],"train_loss":history['train_loss'],
                     'train_miou':history['train_miou'],'val_miou':history['val_miou'],
                     'train_acc':history['train_acc'],'val_acc':history['val_acc']})
    
    df.to_excel("History_resnext101_32x8d_N387_C5_30102023.xlsx")
    
plot_loss(history)
plt.tight_layout()
plt.savefig(r'Plots\\Loss.svg')
# plot_score(history)
# plot_acc(history)

# if __name__ == '__main__':
#     MyCNN()

#%%
import time
from tqdm import tqdm
for i in tqdm(range(100)):
    time.sleep(0.01)
    pass

#%
class DroneTestDataset(Dataset):
    
    def __init__(self, img_path, mask_path, X, transform=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.X = X
        self.transform = transform
      
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.img_path +r'\\'+ self.X[idx] + '.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_path +r'\\'+ self.X[idx] + '.png', cv2.IMREAD_GRAYSCALE)
        
        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = Image.fromarray(aug['image'])
            mask = aug['mask']
        
        if self.transform is None:
            img = Image.fromarray(img)
        
        mask = torch.from_numpy(mask).long()
        
        return img, mask


t_test = A.Resize(416, 416, interpolation=cv2.INTER_NEAREST)
test_set = DroneTestDataset(IMAGE_PATH, MASK_PATH, X_test, transform=t_test)

#%

def predict_image_mask_miou(model, image, mask, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    model.eval()
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    image = t(image)
    model.to(device); image=image.to(device)
    mask = mask.to(device)
    with torch.no_grad():
        
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)
        
        output = model(image)
        score = mIoU(output, mask)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked, score

#%
def predict_image_mask_pixel(model, image, mask, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    model.eval()
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    image = t(image)
    model.to(device); image=image.to(device)
    mask = mask.to(device)
    with torch.no_grad():
        
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)
        
        output = model(image)
        acc = pixel_accuracy(output, mask)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked, acc
#%

image, mask = test_set[3]
pred_mask, score = predict_image_mask_miou(model, image, mask)


def miou_score(model, test_set):
    score_iou = []
    for i in tqdm(range(len(test_set))):
        img, mask = test_set[i]
        pred_mask, score = predict_image_mask_miou(model, img, mask)
        score_iou.append(score)
    return score_iou

#%

mob_miou = miou_score(model, test_set)

def pixel_acc(model, test_set):
    accuracy = []
    for i in tqdm(range(len(test_set))):
        img, mask = test_set[i]
        pred_mask, acc = predict_image_mask_pixel(model, img, mask)
        accuracy.append(acc)
    return accuracy
#%%

mob_acc = pixel_acc(model, test_set)


fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20,10))
ax1.imshow(image)
ax1.set_title('Picture');

ax2.imshow(mask)
ax2.set_title('Ground truth')
ax2.set_axis_off()

ax3.imshow(pred_mask)
ax3.set_title('UNet-MobileNet | mIoU {:.3f}'.format(score))
ax3.set_axis_off()

#%%

image2, mask2 = test_set[62]
pred_mask2, score2 = predict_image_mask_miou(model, image2, mask2)

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(19,6))
ax1.imshow(image2)
ax1.set_title('Picture');

ax2.imshow(mask2)
ax2.set_title('Ground truth')
ax2.set_axis_off()

ax3.imshow(pred_mask2)
ax3.set_title('resnext101_32x8d_N387_C5 | mIoU {:.3f}'.format(score2))
ax3.set_axis_off()
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
fig.tight_layout()
fig.savefig(r"Plots\Class_Test_Acuracy_3_30102023.png",dpi=300)
#%%

image3, mask3 = test_set[16]
pred_mask3, score3 = predict_image_mask_miou(model, image3, mask3)

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20,10))
ax1.imshow(image3)
ax1.set_title('Picture');

ax2.imshow(mask3)
ax2.set_title('Ground truth')
ax2.set_axis_off()

ax3.imshow(pred_mask3)
ax3.set_title('UNet-MobileNet | mIoU {:.3f}'.format(score3))
ax3.set_axis_off()
#%%

print('Test Set mIoU', np.mean(mob_miou))

print('Test Set Pixel Accuracy', np.mean(mob_acc))

#%% Confusion matrix
import seaborn as sns
"""
from: https://www.kaggle.com/code/shrayankm74/semantic-segmentation-is-easy-with-pytorch/notebook
"""
index = 0
# for model in models_loaded:
    
def  ConfusionMat(model,test_set,n_classes=23):
    
    image, mask = test_set[1]
    pred_mask, score = predict_image_mask_miou(model, image, mask)
    
    target = np.array(mask)
    prediction = np.array(pred_mask)
    
    matrix = np.zeros((n_classes, 2))
    rows = target.shape[0]
    cols = target.shape[1]
    
    for i in range(rows):
        for j in range(cols):
            target_pixel = target[i][j]
            predict_pixel = prediction[i][j]
        
            if (target_pixel == predict_pixel):
                matrix[target_pixel][0] = matrix[target_pixel][0] + 1
            else:
                matrix[target_pixel][1] = matrix[target_pixel][1] + 1
        
    max_value, min_value = 0, 0
    for i in range(n_classes):
        for j in range(2):
            max_value = max(max_value, matrix[i][j])
            min_value = min(min_value, matrix[i][j])
    
    for i in range(n_classes):
        for j in range(2):
            matrix[i][j] = (matrix[i][j] - min_value) / (max_value - min_value)
    
    matrix = pd.DataFrame(matrix, columns = ['True', 'False'])
    plt.figure(figsize = (12, 12), dpi = 125)
    sns.heatmap(matrix, annot = True, linewidths=.5);
    
    # plt.ylabel('Classes');
    # plt.xlabel(f'{model_names[index]}')
    # plt.savefig(f'{model_names[index]}_confusion_matrix.jpg', bbox_inches = 'tight')
    
    # index = index + 1
    
    
ConfusionMat(model,test_set,5)

def COM2(model,test_set,n_classes=23):
    image, mask = test_set[1]
    pred_mask, score = predict_image_mask_miou(model, image, mask)
    
    target = np.array(mask)
    prediction = np.array(pred_mask)
    
    matrix = np.zeros((n_classes, n_classes))
    rows = target.shape[0]
    cols = target.shape[1]
    
    for i in range(rows):
        for j in range(cols):
            target_pixel = target[i][j]
            predict_pixel = prediction[i][j]
        
            matrix[predict_pixel][target_pixel] = matrix[predict_pixel][target_pixel] + 1 
        
    max_value, min_value = 0, 0
    for i in range(n_classes):
        for j in range(n_classes):
            max_value = max(max_value, matrix[i][j])
            min_value = min(min_value, matrix[i][j])
    
    for i in range(n_classes):
        for j in range(n_classes):
            matrix[i][j] = (matrix[i][j] - min_value) / (max_value - min_value)
    
    matrix = pd.DataFrame(matrix)
    plt.figure(figsize = (12, 12), dpi = 125)
    sns.heatmap(matrix, annot = True, linewidths=.5);
    # plt.ylabel('Classes');
    # plt.xlabel(f'{model_names[index]}')
    # plt.savefig(f'{model_names[index]}_confusion_matrix_table.jpg', bbox_inches = 'tight')
    # index = index + 1
COM2(model,test_set,5)
