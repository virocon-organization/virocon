import pandas as pd

from pathlib import Path


ROOT_DIR = Path(__file__).parent.parent

def read_ec_benchmark_dataset(file_path=None):
    """
    Reads an EC benchmark dataset.
    
    Reads a text/csv file as it is used in the EC benchmark:
    https://github.com/ec-benchmark-organizers/ec-benchmark

    Parameters
    ----------
    file_path : str
        Path to the dataset file. Defaults to the example dataset A. 
        
    Returns
    -------
    dataset : pandas.DataFrame
        The dataset stored in a pandas DataFrame. 
        Use dataset.values to access the underlying numpy array. 
    """
    if file_path is None:
        file_path = str(ROOT_DIR.joinpath(Path("datasets/ec-benchmark_dataset_A.txt")))
    
    data = pd.read_csv(file_path, sep=";", skipinitialspace=True)
    data.index = pd.to_datetime(data.pop(data.columns[0]), format="%Y-%m-%d-%H")
    return data
    