import pandas as pd
# from collections import Counter
from datetime import datetime, timedelta


def unicos(df):
    for i in df.columns:
        print(i, "\n", df[i].unique(), "\n\n")
    print("Tamaño:", df.shape)

def fecha(data):
    return datetime(1899, 12, 30) + timedelta(data)
