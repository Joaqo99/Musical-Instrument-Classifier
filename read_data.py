import pandas as pd

def read_file(filename):
    """Reads csv file"""
    df = pd.read_csv(filename, sep=";")
    return df