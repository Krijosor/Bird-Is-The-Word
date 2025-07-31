import numpy as np
from PIL import Image
import cv2
import cv2.dnn_superres
import torch
from torch.utils.data import Dataset, DataLoader 
from torchvision.transforms import functional as F
import torchvision.transforms as T
from torchvision.io import decode_image
import BWDatasets
import dfmaker

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
def finalize_image(image, transform=None) -> np.ndarray:

    # Make sure that the ocr image is a tensor
    if not (isinstance(image, torch.Tensor)):
        image = T.ToTensor()(image)

    # Add final transformation mask to image
    if transform:
        image = transform(image)

    # Convert image back to ints and numpy array
    image = BWDatasets.convert_dtype(image)
    image = BWDatasets.tensor_to_numpy(image)
    
    return image

'''
Initializes the bilateral filter
'''
def init_bilateral_filter():
    cv2.createTrackbar("bl_d", "tune", 5, 9, update)
    cv2.createTrackbar("bl_sigmaColor", "tune",  20, 150, update)
    cv2.createTrackbar("bl_sigmaSpace", "tune",  20, 150, update)
    
'''
Function used to apply the Bilateral Filter
'''
def bilateral_filter(image):
    # d = 0 crashes often
    d=cv2.getTrackbarPos('bl_d', 'tune')
    if d == 0:
        d=1

    image = cv2.bilateralFilter(src=image, 
                                d=d, 
                                sigmaColor=cv2.getTrackbarPos('bl_sigmaColor', 'tune'), 
                                sigmaSpace=cv2.getTrackbarPos('bl_sigmaSpace', 'tune'),
                                borderType=cv2.BORDER_DEFAULT)
    #bilat_values = [cv2.BORDER_WRAP, cv2.BORDER_DEFAULT, cv2.BORDER_TRANSPARENT, cv2.BORDER_ISOLATED]
    # image = cv2.bilateralFilter(image, cv2.getTrackbarPos('bl_blockSize', 'tune'), cv2.getTrackbarPos('bl_C', 'tune'), cv2.getTrackbarPos('bl_kernelSize', 'tune'))
    return image

'''
Contains the application that we will use to change the images
'''
def application(dataset):
    # Extract preselected images
    image = dataset[max_n_single]['ocr_image']

    img_W, img_H, _ = image.shape

    cv2.namedWindow("tune", cv2.WINDOW_NORMAL)
    cv2.resizeWindow('tune', 2*img_H, 2*img_W)
    init_bilateral_filter()

    

    while True:
        # Apply filters
        filtered_image = bilateral_filter(image)

        filtered_image = finalize_image(filtered_image)

        # Display Image
        cv2.imshow(winname="tune", mat=filtered_image)

        # Exit with ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cv2.destroyAllWindows()

'''
Shows the images we have filtered with the chosen value
'''
def show_images():
    pass

'''
Refreshes the image when a change to the filter is made
'''
def update(x): pass



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

    max_n_single = 4
    max_n_all = None
    transform = T.Compose([
        # T.Resize((64,32)),
        T.Resize((224,112)),
        T.Lambda(lambda x: F.rotate(x, 270, expand=True))
    ])

    df_lyng = dfmaker.make_dataframe(labels_path=label_path_lyng, img_path=image_path_lyng, bb_path=bb_path_lyng, max_n=max_n_single)
    df_rf = dfmaker.make_dataframe(labels_path=label_path_rf, img_path=image_path_rf, bb_path=bb_path_rf, max_n=max_n_single)
    df_rno = dfmaker.make_dataframe(labels_path=label_path_rno, img_path=image_path_rno, bb_path=bb_path_rno, max_n=max_n_single)
    complete_df = dfmaker.combine_dfs([df_lyng, df_rf, df_rno])

    exp_dataset = BWDatasets.TestDataSet(df=complete_df, max_n=max_n_all)

    application(exp_dataset)









   

