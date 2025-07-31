import numpy as np
from PIL import Image
import cv2
import cv2.dnn_superres
import torch
from torch.utils.data import Dataset, DataLoader 
from torchvision.transforms import functional as F
import torchvision.transforms as T
from torchvision.io import decode_image
from . import BWDatasets
from . import dfmaker

'''

    This file is used to find good values for opencv parameters.

'''



'''
Necessary variables
'''

upres = cv2.dnn_superres.DnnSuperResImpl.create()
upres.readModel('src/upsampling/EDSR_x4.pb')
upres.setModel('edsr', 4)
upres.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
upres.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


'''
Makes an image ready for the pipeline
'''
def finalize_image(image, transform=None) -> torch.Tensor:

    # Make sure that the ocr image is a tensor
    if not (isinstance(image, torch.Tensor)):
        image = T.ToTensor()(image)

    # Add final transformation mask to image
    if transform:
        image = transform(image)

    # Convert image back to ints
    image = BWDatasets.convert_dtype(image)

    return image

'''
Decides the filter to be applied to an image
'''
def apply_filter(image):
    image = ...



    return image


'''
Contains the aplication that we will use to change the images
'''
def application():

    # image = apply_filter(image)
    # image = finalize_image(image)

    pass

'''
Shows the images we have filtered with the chosen value
'''
def show_images():
    pass




if __name__ == "__main__":
    
    # Prepare Dataset
    
    # Lyngøy
    label_path_lyng = "dataset/datasets/lyngoy/ringcodes.csv"
    image_path_lyng = "dataset/datasets/lyngoy/images"
    bb_path_lyng = "dataset/datasets/lyngoy/labels"

    # RF
    label_path_rf = "dataset/datasets/rf/ringcodes.csv"
    image_path_rf = "dataset/datasets/rf/images"
    bb_path_rf = "dataset/datasets/rf/labels"

    # Ringmerkingno
    label_path_rno = "dataset/datasets/ringmerkingno/ringcodes.csv"
    image_path_rno = "dataset/datasets/ringmerkingno/images"
    bb_path_rno = "dataset/datasets/ringmerkingno/labels"

    max_n_single = 400
    max_n_all = None
    transform = T.Compose([
        # T.Resize((64,32)),
        T.Resize((224,112)),
        T.Lambda(lambda x: F.rotate(x, 270, expand=True))
    ])

    df_lyng = dfmaker.make_dataframe(labels_path=label_path_lyng, img_path=image_path_lyng, bb_path=bb_path_lyng, max_n=max_n_single)
    df_rf = dfmaker.make_dataframe(labels_path=label_path_rf, img_path=image_path_rf, bb_path=bb_path_rf, max_n=max_n_single)
    df_rno = dfmaker.make_dataframe(labels_path=label_path_rno, img_path=image_path_rno, bb_path=bb_path_rno, max_n=max_n_single)
    complete_df = dfmaker.combine_dfs([df_lyng, df_rf, df_rno ])

    exp_dataset = BWDatasets.TestDataSet(df=complete_df, transform=transform, max_n=max_n_all)
