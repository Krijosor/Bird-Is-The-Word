import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader 
from torchvision.transforms import functional as F
import torch.nn.functional as Fnn
import torchvision.transforms as T
from torchvision.io import decode_image
import pandas as pd
from PIL import Image
import os
from pathlib import Path
import cv2
from src import dfmaker
#import dfmaker

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
Images that go into dataset are retrieved from the given folder and converted to PIL images. 
These are then converted to tensors.
Then cropped by the Bounding Box coordinates.
Then image is upsampled using opencv2's superresolution module EDSR_x4
Then transform is applied.
'''
class TrainDataSet(Dataset):
    def __init__(self, df:pd.DataFrame, transform=None, max_n=None):
        self.transform = transform
        self.max_n = max_n
        self.clahe = cv2.createCLAHE(clipLimit=15.0, tileGridSize=(7,7))

        # Extract dataframe data
        self.img_paths = df['img_paths'].tolist()
        self.bb_paths = df['bb_paths'].tolist()
        self.labels = df['labels'].tolist()

        # Pre-Calculate bounding box coordinates
        self.bb_cords = []
        for bb, img in zip(self.bb_paths, self.img_paths):
            box = _bb_txt_to_list(bb_path=bb)
            image = decode_image(img)
            self.bb_cords.append(_calculate_bb_cords(image=image, bb=box))
    
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        
        # Retrieve image into a numpy array
        image = decode_image(self.img_paths[idx])

        # Retrieve boundng box coordinates and Crop image
        bb_cords = self.bb_cords[idx]

        # If there is no bounding box then the whole image is processed
        if bb_cords is not None:
            ocr_image = _crop_image_with_bb(image, bb_cords)
        else:
            ocr_image = image

        # Upsample image if it is not too large already
        img_W, img_H, _ = ocr_image.shape
        if img_W < 1000 and img_H < 800:
            ocr_image = ocr_image.to(device=device).float() / 255.
            ocr_image = ocr_image.unsqueeze(0)
            ocr_image = Fnn.interpolate(ocr_image, scale_factor=4.0, mode='bicubic', align_corners=False)
            ocr_image = ocr_image.squeeze(0)
            ocr_image = convert_dtype(ocr_image).cpu()

        ocr_image = tensor_to_numpy(ocr_image)  

        #ocr_image = cv2.cvtColor(ocr_image, cv2.COLOR_RGB2GRAY)

        # Noise reduction
        #ocr_image = cv2.bilateralFilter(ocr_image, 20, 50, 50, borderType=cv2.BORDER_DEFAULT)

        ocr_image = cv2.fastNlMeansDenoisingColored(ocr_image, None, 10, 10, 7, 15)

        # Grayscaling
        # ocr_image = cv2.cvtColor(ocr_image, cv2.COLOR_BGR2RGB)
        # check_image_state(ocr_image, "after grey")
        
        # Normalization with ImageNet mean and std
        # ocr_image = ocr_image.astype(np.float32) / 255.0
        # mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        # std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        # ocr_image = (ocr_image - mean) / std
        # ocr_image = (ocr_image * 255).clip(0, 255).astype('uint8')

        # ocr_image = cv2.cvtColor(ocr_image, cv2.COLOR_RGB2GRAY)

        # Normalization with grayscale, imagenet values
        # ocr_image = ocr_image.astype(np.float32) / 255.0
        # mean = np.array([0.449], dtype=np.float32)
        # std = np.array([0.226], dtype=np.float32)
        # ocr_image = (ocr_image - mean) / std
        # ocr_image = (ocr_image * 255).clip(0, 255).astype('uint8')

        # check_image_state(ocr_image, "before normalization")

        # Contrast normalization variant
        # Either use this or regular normalization
        # dst = np.zeros_like(ocr_image)
        # ocr_image = cv2.normalize(src=ocr_image, dst=dst, alpha=0., beta=255., norm_type=cv2.NORM_MINMAX, dtype=-1, mask=None)
 
        # Adaptive Thresholding
        # ocr_image = cv2.adaptiveThreshold(src=ocr_image, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=9, C=15)

        # Morphological cleanup, requires binary mask
        # Open
        # ocr_image = cv2.morphologyEx(ocr_image, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
        # Close
        # ocr_image = cv2.morphologyEx(ocr_image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))

        # TODO: Data augmentation

        # check_image_state(ocr_image, "before tensor")

        #ocr_image = cv2.cvtColor(ocr_image, cv2.COLOR_GRAY2RGB)

        # Make sure that the image is a tensor
        if not (isinstance(image, torch.Tensor)):
            image = T.ToTensor()(image)

        # Make sure that the ocr image is a tensor
        if not (isinstance(ocr_image, torch.Tensor)):
            ocr_image = T.ToTensor()(ocr_image)

        # Add transformation mask to images
        if self.transform:
            image = self.transform(image)
            ocr_image = self.transform(ocr_image)

        # Convert image back to ints
        ocr_image = convert_dtype(ocr_image)

        # #bb = self.bb_cords[idx]

        return {"ocr_image":ocr_image, "image":image, "label":label}


'''
Copy of TrainDataset used to find good values for the preprocessing filters
'''
class TestDataSet(Dataset):
    def __init__(self, df:pd.DataFrame, max_n=None):
        self.max_n = max_n

        # Extract dataframe data
        self.img_paths = df['img_paths'].tolist()
        self.bb_paths = df['bb_paths'].tolist()
        self.labels = df['labels'].tolist()

        # Pre-Calculate bounding box coordinates
        self.bb_cords = []
        for bb, img in zip(self.bb_paths, self.img_paths):
            box = _bb_txt_to_list(bb_path=bb)
            image = decode_image(img)
            self.bb_cords.append(_calculate_bb_cords(image=image, bb=box))
    
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        
        # Retrieve image into a numpy array
        image = decode_image(self.img_paths[idx])

        # Retrieve boundng box coordinates and Crop image
        bb_cords = self.bb_cords[idx]

        # If there is no bounding box then the whole image is processed
        if bb_cords is not None:
            image = _crop_image_with_bb(image, bb_cords)

        # Upsample
        img_W, img_H, _ = image.shape
        if img_W < 1000 and img_H < 800:
            image = image.to(device=device).float() / 255.
            image = image.unsqueeze(0)
            image = Fnn.interpolate(image, scale_factor=4.0, mode='bicubic', align_corners=False)
            image = image.squeeze(0)
            image = convert_dtype(image).cpu()

        # Convert to numpy, BGR2RGB
        image = tensor_to_numpy(image)   
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return {"ocr_image":image, "label":label}

'''
Converts the image from float values to integer values 
or from integer to float values,
depending on what type the image currently has.

We set copy = False since we won't use the pre-manipulated image again.
'''
def convert_dtype(image:torch.Tensor) -> torch.Tensor:
    image = (image.clamp(0.0, 1.0) * 255).byte()
    return image

'''
Converts a tensor containing a PIL image into a numpy array.
The PaddleOCR model only takes in a numpy array so the image must be converted into one first.
'''
def tensor_to_numpy(img_tensor:torch.Tensor):
    img_np = img_tensor.detach().cpu()
    
    # Make sure dimensionality is taken into account when permuting image dimensions
    if img_tensor.dim() == 4: # For batches
        return img_np.permute(0, 2, 3, 1).numpy()
    
    if img_tensor.dim() == 3: # For single images
        return img_np.permute(1, 2, 0).numpy()
    
    return img_np.numpy()

'''
Converts a txt file containing the bounding boxes to a ring
@param bb_path - The file path leading to a txt file containing the bounding boxes
-------- Helper Function --------
'''
def _bb_txt_to_list(bb_path):
    if bb_path == "No BB Found":
        return None
    else:
        with open(bb_path) as f:
            line = f.readline().strip()
            bb = line.split(' ')
        return bb

'''
Crops an image to the bounding box that is provided
-------- Helper function --------
'''
def _crop_image_with_bb(image, bb):
    x, y, x2, y2 = map(int, bb)
    return image[:, y:y2, x:x2]

'''
Calculates the Bounding box coordinates for an image
@param image - the image's dimensions is needed for box calculations
@param bb - the bounding box has the following: [x_middle, y_middle, width, height]
-------- Helper Function --------
'''
def _calculate_bb_cords(image, bb):
    # If bounding box does not exist, exit
    if bb is None:
        return None

    # img_H, img_W, _ = image.shape
    _, img_H, img_W= image.shape

    # Calculate x and y coordinates of the bb
    bb_x = img_W * float(bb[1])
    bb_y = img_H * float(bb[2])

    # Calculate height and width of the bb, and divide by 2
    bb_w = (img_W * float(bb[3])) / 2
    bb_h = (img_H * float(bb[4])) / 2

    # Calculate corners of bb
    min_x = max(0, int(bb_x - (bb_w)))
    max_x = min(img_W, int(bb_x + (bb_w)))
    min_y = max(0, bb_y - int((bb_h)))
    max_y = min(img_H, bb_y + int((bb_h)))

    return min_x, min_y, max_x, max_y

'''
Used to check the state of an image during the retrieval
'''
def check_image_state(img, state):
    print(f'Checking image at state: {state}')
    print(f'Is image None: {img is None}')
    print(f'Shape: {img.shape}, Pixel Datatype: {img.dtype}')
    print(f'Min Value: {img.min()}, Max Value: {img.max()}')
    print("-"*12)

'''
Main function for file
Used for testing, mainly that the preprocessing steps keep the desired output format

'''
if __name__ == "__main__":

    label_path = "dataset/datasets/rf/ringcodes.csv"
    image_path = "dataset/datasets/rf/images"
    bb_path = "dataset/datasets/rf/labels"
    max_n = 1
    WIDTH = 224
    HEIGHT = 224

    transform = T.Compose([
        T.Resize((HEIGHT,WIDTH), antialias=True),
        T.Lambda(lambda x: F.rotate(x, 270, expand=True))
        
    ])

    df = dfmaker.make_dataframe(img_path=image_path, labels_path=label_path, bb_path=bb_path, max_n=max_n)

    train_dataset = TrainDataSet(df=df,transform=transform, max_n=max_n)

    exp_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

    for data in exp_loader:
        images = data["ocr_image"]
        for image in images:
            image = tensor_to_numpy(image)
            # Some simple tests to ensure image correctness
            if image is None:
                print("Image is None!")
            elif image.dtype.name == 'float32':
                print("Image is a float!")
            elif image.min() < 0:
                print("Image minimum value is lower than 0!")
            elif image.max() <= 2:
                print("Image maximum value is too low!")
            else:
                print("All Good!")