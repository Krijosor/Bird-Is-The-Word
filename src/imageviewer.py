import numpy as np
import cv2
import cv2.dnn_superres
import torch
from torchvision.transforms import functional as F
import torchvision.transforms as T
import BWDatasets
import dfmaker

'''

    This file is used to find good values for opencv parameters.

'''

'''
Necessary variables
'''
# Upsampler
upres = cv2.dnn_superres.DnnSuperResImpl.create()
upres.readModel('src/upsampling/EDSR_x4.pb')
upres.setModel('edsr', 4)
upres.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
upres.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Bilateral Filter Variables
bilat_values = [cv2.BORDER_WRAP, cv2.BORDER_DEFAULT, cv2.BORDER_TRANSPARENT, cv2.BORDER_ISOLATED]

# Contrast Enhancer Variables
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(20,20))

# Adaptive Thresholding Variables
adaptive_bsizes = [1, 3, 5, 7, 9, 11, 13, 15]
adaptive_method = [cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.ADAPTIVE_THRESH_MEAN_C]
adaptive_thresh_type = [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV]

# Morphological Cleanup Variables
mc_shape = [cv2.MORPH_RECT, cv2.MORPH_CROSS, cv2.MORPH_ELLIPSE]
mc_kernel_size = [1, 3, 5, 7, 9, 11, 13, 15]

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
    cv2.createTrackbar("bl_d", "tuned", 5, 15, update)
    cv2.createTrackbar("bl_sigmaColor", "tuned", 50, 150, update)
    cv2.createTrackbar("bl_sigmaSpace", "tuned",  50, 150, update)
    cv2.createTrackbar("bl_borderType", "tuned",  0, len(bilat_values) - 1, update)

    cv2.setTrackbarMin('bl_d', 'tuned', 1)
    cv2.setTrackbarMin('bl_sigmaColor', 'tuned', 1)
    cv2.setTrackbarMin('bl_sigmaSpace', 'tuned', 1)
    
'''
Apply the Bilateral Filter to an image
'''
def bilateral_filter(image):
    btype = bilat_values[cv2.getTrackbarPos('bl_borderType', 'tuned')]
    image = cv2.bilateralFilter(src=image, 
                                d=cv2.getTrackbarPos('bl_d', 'tuned'), 
                                sigmaColor=cv2.getTrackbarPos('bl_sigmaColor', 'tuned'), 
                                sigmaSpace=cv2.getTrackbarPos('bl_sigmaSpace', 'tuned'),
                                borderType=btype)
    return image

'''
Initializes the Contrast Enhancement
'''
def init_contrast_enchancement():
    cv2.createTrackbar('ce_limit', 'tuned', 40, 150, update)
    cv2.createTrackbar('ce_size', 'tuned', 4, 20, update)

    cv2.setTrackbarMin('ce_limit', 'tuned', 1)
    cv2.setTrackbarMin('ce_size', 'tuned', 2)

'''
Apply the Contrast Enhancement Filter to an image
'''
def contrast_enhancement(image):
    clahe.setClipLimit(cv2.getTrackbarPos('ce_limit', 'tuned'))
    clahe.setTilesGridSize((cv2.getTrackbarPos('ce_size', 'tuned'), cv2.getTrackbarPos('ce_size', 'tuned')))
    image = clahe.apply(image)
    return image

'''
Apply gray scale filter to image
'''
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

'''
Apply 3d standardization with imagenet values
'''
def standardization_3d(image):
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std
    image = (image * 255).clip(0, 255).astype('uint8')
    return image

'''
Apply 1d standardization with imagenet values
'''
def standardization_1d(image):
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.449], dtype=np.float32)
    std = np.array([0.226], dtype=np.float32)
    image = (image - mean) / std
    image = (image * 255).clip(0, 255).astype('uint8')
    return image

'''
Initializes the Contrast Normalization
Cv2.MinMax is the only norm type used
'''
def init_contrast_normalization():
    cv2.createTrackbar('cn_alpha', 'tuned', 20, 20, update)
    cv2.createTrackbar('cn_beta', 'tuned', 0, 20, update)

    cv2.setTrackbarMin('cn_alpha', 'tuned', 2)

'''
Apply Contrast Normalization
'''
def contrast_normalization(image):
    dst = np.zeros_like(image)
    alpha = float(cv2.getTrackbarPos('cn_alpha', 'tuned')) / 20.
    beta = float(cv2.getTrackbarPos('cn_beta', 'tuned')) / 20.
    image = cv2.normalize(src=image, dst=dst, alpha=alpha, beta=beta, norm_type=cv2.NORM_MINMAX, dtype=-1, mask=None)
    return image

'''
Initialize Adaptive Thresholding
'''
def init_adaptive_thresholding():
    cv2.createTrackbar('at_method', 'tuned', 0, len(adaptive_method) - 1, update)
    cv2.createTrackbar('at_threshType', 'tuned', 0, len(adaptive_thresh_type) - 1, update)
    cv2.createTrackbar('at_bsize', 'tuned', 3, len(adaptive_bsizes) - 1, update)
    cv2.createTrackbar('at_c', 'tuned', 50, 240, update)

    cv2.setTrackbarMin('at_bsize', 'tuned', 1)
    cv2.setTrackbarMin('at_c', 'tuned', 1)

'''
Apply Adaptive Thresholding to an image
'''
def adaptive_thresholding(image):
    bsize = adaptive_bsizes[cv2.getTrackbarPos('at_bsize', 'tuned')]
    method = adaptive_method[cv2.getTrackbarPos('at_method', 'tuned')]
    threshtype = adaptive_thresh_type[cv2.getTrackbarPos('at_threshType', 'tuned')]
    image = cv2.adaptiveThreshold(src=image, maxValue=255, 
                                  adaptiveMethod=method, 
                                  thresholdType=threshtype, 
                                  blockSize=bsize, 
                                  C=cv2.getTrackbarPos('at_c', 'tuned'))
    return image

'''
Initialize Morphological Cleanup

Borders are inconsequential for our task so we do not need parameters for them.
'''
def init_morphological_cleanup():
    cv2.createTrackbar('mc_shape', 'tuned', 0, len(mc_shape) - 1, update)
    cv2.createTrackbar('mc_size_open', 'tuned', 1, len(mc_kernel_size) - 1, update)
    cv2.createTrackbar('mc_size_close', 'tuned', 1, len(mc_kernel_size) - 1, update)

    cv2.setTrackbarMin('mc_size_open', 'tuned', 1)
    cv2.setTrackbarMin('mc_size_close', 'tuned', 1)

'''
Apply Morphological cleanup to image
'''
def morphological_cleanup(image):
    shape = mc_shape[cv2.getTrackbarPos('mc_shape', 'tuned')]
    size_open = mc_kernel_size[cv2.getTrackbarPos('mc_size_open', 'tuned')]
    size_close = mc_kernel_size[cv2.getTrackbarPos('mc_size_close', 'tuned')]

    # Open
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, 
                             cv2.getStructuringElement(shape, (size_open, size_open)))
    # Close
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, 
                             cv2.getStructuringElement(shape, (size_close, size_close)))
    return image

'''
Apply the selected filters to an image and return it
'''
def apply_filters(image):

    filtered_image = bilateral_filter(image)

    # filtered_image = contrast_enhancement(filtered_image)

    # filtered_image = grayscale(filtered_image)

    # filtered_image = standardization_1d(filtered_image)
    
    # filtered_image = standardization_3d(filtered_image)

    # filtered_image = contrast_normalization(filtered_image)

    # filtered_image = adaptive_thresholding(filtered_image)

    # filtered_image = morphological_cleanup(filtered_image)

    return filtered_image

'''
Contains the application that we will use to change the images
'''
def application(dataset):
    # Extract preselected images
    image = dataset[max_n_single+1]['ocr_image']

    # Init image display windows
    img_W, img_H, _ = image.shape
    cv2.namedWindow("tuned", cv2.WINDOW_NORMAL)
    cv2.resizeWindow('tuned', 2*img_W, 2*img_H)
    cv2.namedWindow("regular", cv2.WINDOW_NORMAL)
    cv2.resizeWindow('regular', 2*img_W, 2*img_H)
    cv2.moveWindow('tuned', 100, 100)
    cv2.moveWindow('regular', 600, 100)
    
    # Init the filters
    init_bilateral_filter()

    # Application start
    while True:
        # Apply filters
        filtered_image = apply_filters(image)

        # Display Image
        cv2.imshow(winname="tuned", mat=finalize_image(filtered_image, transform=transform))
        cv2.imshow(winname="regular", mat=finalize_image(image, transform=transform))

        # Exit the application with ESC key
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
Currently empty
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









   

