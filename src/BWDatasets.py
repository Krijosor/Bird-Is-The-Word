import torch
import numpy
from torch.utils.data import Dataset, DataLoader 
import torchvision.transforms as T
import pandas as pd
from PIL import Image
import os

class TrainDataSet(Dataset):
    def __init__(self, img_path:str, labels_path:str, transform=None, max_n=None):
        self.img_path = img_path
        self.transform = transform
        self.max_n = max_n

        # Retrieve labels and sort them in alphabetical order to match the images
        df = pd.read_csv(labels_path, sep="|")
        df = df.sort_values("filename", ascending=True).reset_index(drop=True)

        # Ensure the data contains the chosen amount of elements
        if max_n is not None:
            df = df[:self.max_n]
        
        # Retrieve images from folder and match them with labels
        self.img_paths = df["filename"].apply(lambda e: os.path.join(img_path, e)).to_list()

        self.labels = list(df[["code","color"]].itertuples(index=False, name=None)) # Labels should be a list of tuples
        
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        image = Image.open(self.img_paths[idx]).convert('RGB')

        if self.transform: # Add transformation mask to image
            image = self.transform(image)
            # image = tensor_to_numpy(image) # Image must be a numpy array
        
        return image, label
    
class InferenceDataSet(Dataset):
    def __init__(self, img_path:str, boundingbox_path:str, transform=None, max_n=None):
        self.img_path = img_path
        self.transform = transform
        self.max_n = max_n

        # Retrieve labels and sort them in alphabetical order to match the images
        df = pd.read_csv(boundingbox_path, sep="|")
        df = df.sort_values("filename", ascending=True).reset_index(drop=True)

        # Ensure the data contains the chosen amount of elements
        if max_n is not None:
            df = df[:self.max_n]
        
        # Retrieve images from folder and match them with labels
        self.img_paths = df["filename"].apply(lambda e: os.path.join(img_path, e)).to_list()

        self.labels = list(df[["code","color"]].itertuples(index=False, name=None)) # Labels should be a list of tuples
        
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        image = Image.open(self.img_paths[idx]).convert('RGB')

        if self.transform: # Add transformation mask to image
            image = self.transform(image)
            image = tensor_to_numpy(image) # Image must be a numpy array
        
        return image, label

# Helper function
def tensor_to_numpy(img_tensor):
    img_np = img_tensor.detach().cpu()
    
    if img_tensor.dim() == 4:
        return img_np.permute(0, 3, 2, 1).numpy()
    
    if img_tensor.dim() == 3:
        return img_np.permute(1, 2, 0).numpy()
    # #img_np= (img_np* 255).clip(0, 255).astype('uint8')
    return img_np.numpy()
    

def open_image(filepath):
    image = Image.open(filepath).convert('RGB')
    func = T.PILToTensor()
    image = func(image)
    image = tensor_to_numpy(image)
    return image

if __name__ == "__main__":
    label_path = "dataset/datasets/rf/ringcodes.csv"
    image_path = "dataset/datasets/rf/images"
    max_n = 10
    transform = T.Compose([
        #T.Resize((224,224)),
        T.ToTensor()
    ])
    train_dataset = TrainDataSet(img_path=image_path, labels_path=label_path, transform=transform, max_n=max_n)

    exp_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

    for images, labels in exp_loader:
        print(images.shape)
        images = tensor_to_numpy(images)
        print(images.shape)
        for image in images:
            print(image.shape)


