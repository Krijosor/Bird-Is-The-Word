import os
from pathlib import Path
import pandas as pd

'''
Makes a dataframe 

Args:
    img_path (str): the relative system path leading to the images
    labels_path (str): the relative system path leading to the labels for the images
        - should be in a csv file separated by '|'
    bb_path (str): the relative system path leading to the bounding box coordinates contained within the images
    max_n (int | None): Option to set a limit on how many elements are added to the dataframe. Set max_n=None for all elements to be added.

Returns:
    pd.DataFrame: a sorted pandas dataframe containing the columns:
        "filename"  -> name of the image file, 
        "img_paths" -> path of the image file, 
        "bb_paths"  -> path to the file containing the bounding boxes,
        "labels"    -> a tuple containing the code and color of the birds bracelet
'''
def make_dataframe(img_path:str, labels_path:str, bb_path: str, max_n:int | None) -> pd.DataFrame:

    df = pd.read_csv(labels_path, sep="|")
    df = df.sort_values("filename", ascending=True).reset_index(drop=True)

    # Ensure the data contains the chosen amount of elements
    if max_n is not None:
        df = df[:max_n]
    
    # Retrieve images from folder and match them with labels
    df['img_paths'] = df["filename"].apply(lambda filename: os.path.join(img_path, filename))
    df['bb_paths'] = df["filename"].apply(lambda filename: os.path.join(bb_path, filename[:-4] + ".txt"))
    df['labels'] = list(df[["code","color"]].itertuples(index=False, name=None))

    # Validate files to make sure they are ok
    df['img_paths'] = df["img_paths"].apply(lambda item: validate_path(item))
    df['bb_paths'] = df["bb_paths"].apply(lambda item: validate_path(item))

    # These columns are no longer needed
    df.drop(columns=["color", "code"])
    return df

'''
Combines multiple dataframes into a single dataframe.

Args:
    df_list (list[pd.DataFrame]): A list of pandas dataframes

Returns:
    a sorted dataframe containing all the dataframes in the list
'''
def combine_dfs(df_list:list[pd.DataFrame]) -> pd.DataFrame:
    complete_df = pd.DataFrame(columns=['filename', 'img_paths', 'bb_paths'])

    for df in df_list:
        complete_df = pd.concat([complete_df, df], ignore_index=True, axis=0)

    return complete_df.sort_values("filename", ascending=True).reset_index(drop=True)

'''
A function that validates a given path.
If the path is for an image it raises an exception if it is not found.
If the path is for a bounding box it replaces the None value of the bounding box with "No BB Found".
This way, if no bb is found for an image, we still try with the whole image and hope for the best.

Args:
    path (str): the path as a string

Returns:
    str: The umodified image path, or a marked bb path if no bb is found

Raises:
    an Exception if no image is found for the image path
'''
def validate_path(path:str) -> str:
    syspath = Path(path)
    if not syspath.exists():
        # If a bounding box does not exist, feed the whole image
        if path[-4:] == ".txt":
            path = "No BB Found"
        else:
            raise Exception(f'Path {syspath} does not exist.')
        
    return path

if __name__ == '__main__':
    
    # Simple check for file

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

    df_lyng = make_dataframe(img_path=image_path_lyng, labels_path=label_path_lyng, bb_path=bb_path_lyng, max_n=None)
    df_rf = make_dataframe(img_path=image_path_rf, labels_path=label_path_rf, bb_path=bb_path_rf, max_n=None)
    df_rno = make_dataframe(img_path=image_path_rno, labels_path=label_path_rno, bb_path=bb_path_rno, max_n=100)

    combined_df = combine_dfs([df_lyng, df_rf, df_rno])

    lyn_len = len(df_lyng)
    rf_len = len(df_rf)
    rno_len = len(df_rno)
    combined_len = len(combined_df)

    if combined_len == lyn_len + rf_len + rno_len:
        print('All Good')
    else:
        print('Lengths are dissimiliar')


