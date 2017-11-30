import numpy as np
import pandas as pd
from sklearn import preprocessing

def preprocess(edit):
    edit.replace('?',np.NaN)
    edit = edit[["age","education-num","capital-gain","capital-loss","hours-per-week"]]
    edit = edit.dropna()
    edit = preprocessing.normalize(edit)
    return edit

if __name__ == "__main__":
    data = pd.read_csv('censusincome.csv')
    new_data = preprocess(data)
    print(new_data)

