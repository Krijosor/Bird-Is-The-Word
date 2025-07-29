import os
from pathlib import Path
import pandas as pd

'''
Makes a dataframe 

'''
def make_dataframe(img_path:str, labels_path:str, bb_path: str, max_n:int) -> pd.DataFrame:

    df = pd.read_csv(labels_path, sep="|")
    df = df.sort_values("filename", ascending=True).reset_index(drop=True)

    # Ensure the data contains the chosen amount of elements
    if max_n is not None:
        df = df[:max_n]
    
    # Retrieve images from folder and match them with labels
    #img_paths = df["filename"].apply(lambda filename: os.path.join(img_path, filename))
    #bb_paths = df["filename"].apply(lambda filename: os.path.join(bb_path, filename[:-4] + ".txt"))
    #labels = df[["code","color"]].itertuples(index=False, name=None) # Labels are tuples 

    df['img_paths'] = df["filename"].apply(lambda filename: os.path.join(img_path, filename))
    df['bb_paths'] = df["filename"].apply(lambda filename: os.path.join(bb_path, filename[:-4] + ".txt"))
    df['labels'] = df[["code","color"]].itertuples(index=False, name=None)

    # Validate files to make sure they are ok
    # img_paths = validate_paths(df['img_paths'].tolist())
    # bb_paths = validate_paths(df['bb_paths'].tolist())

    return df

'''
Combines multiple dataframes into a single dataframe
'''
def combine_dfs(df_list):
    complete_df = pd.DataFrame(columns=['filename', 'img_paths', 'bb_paths'])

    for df in df_list:
        complete_df = pd.concat([complete_df, df], ignore_index=True, axis=0)

    complete_df = complete_df.sort_values("filename", ascending=True).reset_index(drop=True)

    return complete_df


# Validates that each file path actually exists
# Throw an exception if a path does not exist
def validate_paths(paths):
    for i, path in enumerate(paths):
        syspath = Path(path)
        if not syspath.exists():
            # If a bounding box does not exist, feed the whole image
            if path[-4:] == ".txt":
                paths[i] = "No BB"
            else:
                raise Exception(f'Path {syspath} does not exist.')
            
    return paths


if __name__ == '__main__':
    

    # Noen tester

    pass




