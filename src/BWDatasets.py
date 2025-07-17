import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader 
from torchvision.transforms import functional as F
import torchvision.transforms as T
import pandas as pd
from PIL import Image
import os
import cv2
import cv2.dnn_superres

'''
Images that go into dataset are retrieved from the given folder and converted to PIL images. 
These are then converted to tensors.
Then cropped by the Bounding Box coordinates.
Then image is upsampled using opencv2's superresolution module EDSR_x4
Then transform is applied.
'''
class TrainDataSet(Dataset):
    def __init__(self, img_path:str, labels_path:str, bb_path: str, transform=None, max_n=None):
        self.img_path = img_path
        self.transform = transform
        self.max_n = max_n
        self.upres = cv2.dnn_superres.DnnSuperResImpl.create()
        self.upres.readModel('src/upsampling/EDSR_x4.pb')
        self.upres.setModel('edsr', 4)

        # Retrieve labels and sort them in alphabetical order to match the images
        df = pd.read_csv(labels_path, sep="|")
        df = df.sort_values("filename", ascending=True).reset_index(drop=True)

        # Ensure the data contains the chosen amount of elements
        if max_n is not None:
            df = df[:self.max_n]
        
        # Retrieve images from folder and match them with labels
        self.img_paths = df["filename"].apply(lambda e: os.path.join(img_path, e)).to_list()

        bb_path = bb_path + "/"

        bb_paths = df["filename"].apply(lambda e: os.path.join(bb_path, e)).to_list()

        self.bb_paths = [i[:-4] + ".txt" for i in bb_paths]

        self.labels = list(df[["code","color"]].itertuples(index=False, name=None)) # Labels should be a list of tuples
        
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        bb_path = self.bb_paths[idx]
        image = Image.open(self.img_paths[idx]).convert('RGB')

        # Make sure that the image is a tensor, and not PIL
        if not (isinstance(image, torch.Tensor)):
            image = T.ToTensor()(image)
        
        bb = _calculate_bb_cords(image=image, bb=_bb_txt_to_list(bb_path=bb_path)) # Retrieve bb cords directly before returning
        image = _crop_image_with_bb(image, bb)

        image = tensor_to_numpy(image) # CV2 requires numpy

        # OpenCV model requires Integers and will truncate our floats, so we convert them safely with this function
        image = convert_dtype(image)

        image = self.upres.upsample(image)

        # Convert values back to floats
        image = convert_dtype(image)

        # Make sure that the image is a tensor
        if not (isinstance(image, torch.Tensor)):
            image = T.ToTensor()(image)

        # Add transformation mask to image
        if self.transform:
            image = self.transform(image)

        return image, bb, label

'''
Converts the image from float values to integer values 
or from integer to float values,
depending on what type the image currently has.

We set copy = False since we won't use the pre-manipulated image again.
'''
def convert_dtype(image):
    
    if image.dtype.name == 'int8':
        image = image.astype(np.float32, copy=False) / 255.0
        image[image > 1.0] = 1.0
        image[image < 0.0] = 0.0

    elif image.dtype.name == 'float32':
        image *= 255
        image = image.astype(np.uint8, copy=False)
        image[image> 255] = 255
        image[image < 0] = 0

    return image

'''
Converts a tensor containing a PIL image into a numpy array.
The PaddleOCR model only takes in a numpy array so the image must be converted into one first.
'''
def tensor_to_numpy(img_tensor:torch.Tensor):
    img_np = img_tensor.detach().cpu()
    
    # Make sure dimensionality is taken into account when permuting image dimensions
    if img_tensor.dim() == 4: # For batches
        return img_np.permute(0, 3, 2, 1).numpy()
    
    if img_tensor.dim() == 3: # For single images
        return img_np.permute(1, 2, 0).numpy()
    
    return img_np.numpy()
    
'''
Opens a single image so that it can easily be converted into
a format that can go into the PaddleOCR model
'''
def open_image(filepath):
    image = Image.open(filepath).convert('RGB')
    func = T.PILToTensor()
    image = func(image)
    image = tensor_to_numpy(image)
    return image

'''
Crops an image to the bounding box that is provided
-------- Helper function --------
'''
def _crop_image_with_bb(image, bb):
    x = int(bb[0])
    y = int(bb[1])
    width = int(bb[2]) - int(bb[0])
    height = int(bb[3]) - int(bb[1])

    img_crop = F.crop(image, top=y, left=x, height=height, width=width)
    return img_crop


'''
Converts a txt file containing the bounding boxes to a ring
@param bb_path - The file path leading to a txt file containing the bounding boxes
-------- Helper Function --------
'''
def _bb_txt_to_list(bb_path):

    with open(bb_path) as f:
        line = f.readline().strip()
        bb = line.split(' ')
    
    return bb

'''
Calculates the Bounding box coordinates for an image
@param image - the image's dimensions is needed for box calculations
@param bb - the bounding box has the following: [x_middle, y_middle, width, height]
-------- Helper Function --------
'''
def _calculate_bb_cords(image, bb):

    img_w = image.shape[2]
    img_h = image.shape[1]

    # Calculate x and y coordinates of the bb
    bb_x = img_w * float(bb[1])
    bb_y = img_h * float(bb[2])

    # Calculate height and width of the bb, and divide by 2
    bb_w = (image.shape[2] * float(bb[3])) / 2
    bb_h = (image.shape[1] * float(bb[4])) / 2

    # Calculate corners of bb
    min_x = max(0, bb_x - (bb_w))
    max_x = min(img_w, bb_x + (bb_w))
    min_y = max(0, bb_y - (bb_h))
    max_y = min(img_h, bb_y + (bb_h))

    return min_x, min_y, max_x, max_y

'''
Used to check the state of an image during the retrieval
'''
def check_image_state(img, state):
    print("-"*12)
    print(f'Checking image at state: {state}')
    print(f'Is image None: {img is None}')
    print(f'Shape: {img.shape}, Pixel Datatype: {img.dtype}')
    print(f'Min Value: {img.min()}, Max Value: {img.max()}')
    print("-"*12)

'''
Main function for file
Used for testing

'''
if __name__ == "__main__":
    label_path = "dataset/datasets/rf/ringcodes.csv"
    image_path = "dataset/datasets/rf/images"
    bb_path = "dataset/datasets/rf/labels"
    max_n = 10
    transform = T.Compose([
        #T.Resize((224,224)),
        T.ToTensor()
    ])
    train_dataset = TrainDataSet(img_path=image_path, labels_path=label_path, bb_path=bb_path, transform=transform, max_n=max_n)

    exp_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

    for images, labels in exp_loader:
        print(images.shape)
        images = tensor_to_numpy(images)
        print(images.shape)
        for image in images:
            print(image.shape)


