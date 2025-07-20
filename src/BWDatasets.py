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
        # image = cv2.imread(self.img_paths[idx], flags=cv2.IMREAD_COLOR_BGR)
        image = np.array(image)
        image = image[:, :, ::-1].copy()

        # Make sure that the image is a tensor, and not PIL
        # if not (isinstance(image, torch.Tensor)):
        #     image = T.ToTensor()(image)
        
        bb = _calculate_bb_cords(image=image, bb=_bb_txt_to_list(bb_path=bb_path)) # Retrieve bb cords directly before returning
        image = _crop_image_with_bb(image, bb)


        # image = T.Normalize(           # ImageNet stats
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225],
        # )(image)

        # Cast to numpy and change to integers
        # image = tensor_to_numpy(image)
        # image = (image * 255).clip(0, 255).astype('uint8')

        # Convert image to OpenCV BGR format, supersample, and convert back to RGB 
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        #image = self.upres.upsample(image)


        # TODO: Deskew bildet

        # TODO: Grayscale bildet

        # TODO: Noise reduction og sharpening -> bilateral filter beholder edges

        # TODO: Contrast enhancement - CLAHE

        # TODO: Binarize / threshold bildet -> en svart og hvit maske

        # TODO: Morphological cleanup -> fyll inn manglende deler av bokstaver og fjern flekker
        

        # TODO: Data augmentation

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Make sure that the image is a tensor
        if not (isinstance(image, torch.Tensor)):
            image = T.ToTensor()(image)

        # Add transformation mask to image
        if self.transform:
            image = self.transform(image)

        # Convert image back to ints
        image = convert_dtype(image)

        return image, bb, label

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
Opens a single image so that it can easily be converted into
a format that can go into the PaddleOCR model
'''
def open_image(filepath):
    image = Image.open(filepath).convert('RGB')
    func = T.PILToTensor()
    image = func(image)
    # TODO: Add preprocessing steps
    image = tensor_to_numpy(image)
    return image

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
Crops an image to the bounding box that is provided
-------- Helper function --------
'''
def _crop_image_with_bb(image, bb):

    x, y, x2, y2 = map(int, bb)

    # x = int(bb[0])
    # y = int(bb[1])
    # x2 = int(bb[2])
    # y2  = int(bb[3])
    # width = int(bb[2]) - int(bb[0])
    # height = int(bb[3]) - int(bb[1])

    # x2 = x + width
    # y2 = y + height

    # x, y, x2, y2 = map(int, bb)
    return image[y:y2, x:x2, :]

# def _crop_image_with_bb(image, bb):
#     x = int(bb[0])
#     y = int(bb[1])
#     width = int(bb[2]) - int(bb[0])
#     height = int(bb[3]) - int(bb[1])

#     img_crop = F.crop(image, top=y, left=x, height=height, width=width)
#     return img_crop

'''
Calculates the Bounding box coordinates for an image
@param image - the image's dimensions is needed for box calculations
@param bb - the bounding box has the following: [x_middle, y_middle, width, height]
-------- Helper Function --------
'''
def _calculate_bb_cords(image, bb):

    img_H, img_W, _ = image.shape

    # Calculate x and y coordinates of the bb
    bb_x = img_W * float(bb[1])
    bb_y = img_H * float(bb[2])

    # Calculate height and width of the bb, and divide by 2
    bb_w = (img_W * float(bb[3])) / 2
    bb_h = (img_H * float(bb[4])) / 2

    # Calculate corners of bb
    min_x = max(0, bb_x - (bb_w))
    max_x = min(img_W, bb_x + (bb_w))
    min_y = max(0, bb_y - (bb_h))
    max_y = min(img_H, bb_y + (bb_h))

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
Used for testing, mainly that the preprocessing steps keep the desired output format

'''
if __name__ == "__main__":
    #print(cv2.haveImageReader(".jpg")) 
    
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

    train_dataset = TrainDataSet(img_path=image_path, labels_path=label_path, bb_path=bb_path, transform=transform, max_n=max_n)

    exp_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

    for images, bb, labels in exp_loader:
        images = tensor_to_numpy(images)
        for image in images:
            # Some simple tests
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