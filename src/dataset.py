import os
import numpy as np
from PIL import Image
from scipy.io import loadmat
from tqdm import tqdm
from sklearn.model_selection import train_test_split as tts
from keras.utils import to_categorical
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class data_set(Dataset):
    def __init__(self, img, label, transform=None):
        self.img = img
        self.label = label
        self.transform = transform
        
    def __getitem__(self, idx):
        img = self.img[idx]
        label = self.label[idx]
        if self.transform:
            img = self.transform(img)
        return img, label
        
    def __len__(self):
        return len(self.label)

def load_data(image_folder="data/jpg", label_file="data/imagelabels.mat"):
    all_imgs = []
    all_labels = []
    
    for img_name in tqdm(os.listdir(image_folder)):
        image_path = os.path.join(image_folder, img_name)
        img = Image.open(image_path)
        img_array = np.array(img)
        all_imgs.append(img_array)
        all_labels.append(int(img_name[6:11])-1)
        
    img_labels = loadmat(label_file)
    img_labels = img_labels['labels'][0]
    
    # Subtract 1 from all label values to start from 0
    all_labels = [img_labels[i] - 1 for i in all_labels] 
    all_labels = to_categorical(all_labels, num_classes=102, dtype=float)
    
    return all_imgs, all_labels

def get_dataloaders(all_imgs, all_labels, batch_size=128):
    trainx, testx, trainy, testy = tts(all_imgs, all_labels, test_size=0.2, random_state=7)
    valx, testx, valy, testy = tts(testx, testy, test_size=0.5, random_state=7)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), interpolation=Image.BILINEAR, antialias=None)
    ])

    train_transform = transforms.Compose([
        transform,
        transforms.RandomHorizontalFlip(p=0.1),
        transforms.RandomRotation(degrees=10),
    ])

    train_set = data_set(trainx, trainy, transform=train_transform)
    val_set = data_set(valx, valy, transform=transform)
    test_set = data_set(testx, testy, transform=transform)

    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
