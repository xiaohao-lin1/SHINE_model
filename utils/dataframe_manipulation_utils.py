import pandas as pd
def save_results_to_pdseries(**kwargs):
    index_ls = []
    data_ls = []
    for (k, v) in kwargs.items():
        index_ls.append(k)
        data_ls.append(v)
    return pd.Series(data_ls, index=index_ls)

