import pandas as pd
import os

BASE_PATH = "example_csv_data/"


def load_csv(file_path):
    """
    Load csv file as a pandas DataFrame.

    :param file_path: str - The path to the CSV file.
    :return: pd.DataFrame - A DataFrame containing the data from the specified CSV file.
    """
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        raise
    except Exception as e:
        print(f"An error occurred while loading the CSV file: {e}")
        raise
def get_row_value(data, row_number, column_name=None):
    """
    Get a specific value from a row in a pandas DataFrame."
    """
    try:
        if column_name is None:
            return data.iloc[row_number]
        else:
            return data.at[row_number, column_name]
    except IndexError:
        print(f"Error: Row number {row_number} does not exist in the DataFrame.")
        raise
    except KeyError:
        print(f"Error: Column '{column_name}' does not exist in the DataFrame.")
        raise


def main():
    # Example usage of load_csv function
    csv_file = os.path.join(BASE_PATH, "found_files/export_of_online_json.csv")
    df = load_csv(csv_file)
    if not df.empty:
        print(df.head())  # Print the first few rows of the DataFrame
        print(df.columns)  # Print the column names of the DataFrame
        print(get_row_value(df, 0, df.columns[2]))  # Get a specific value from the first row


if __name__ == "__main__":
    main()
