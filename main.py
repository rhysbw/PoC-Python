import pandas as pd
import os

BASE_PATH = "example_csv_data/"


def load_csv(file_path):
    """
    Load csv file as pandas dataframe

    :param file_path: str: path to csv file
    :return: pd.DataFrame: dataframe with csv data
    """
    return pd.read_csv(file_path)

def main():
    # join the path to the csv file
    path = os.path.join(BASE_PATH, "found_files/export_of_online_json.csv")

    # get a pandas dataframe
    df = load_csv(path)  
    print(df)


if __name__ == "__main__":
    main()