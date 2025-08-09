from src import BWDatasets
import matplotlib.pyplot as plt

def draw_image(img):
    img_uint8 = BWDatasets.tensor_to_numpy(img)
    plt.figure(figsize=(12,12))
    plt.imshow(img_uint8)
    plt.axis("off")
    plt.show()

def find_width(height):
    return int(1.7*height) # 1.7 worked best for the provided dataset




