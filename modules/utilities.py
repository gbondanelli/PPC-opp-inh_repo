import pandas as pd
from ast import literal_eval

# to convert a string of the form '[1   2\n 4.5 3]' into list [1,2,4.5,3]
def my_literal_eval(str):
    str = str.replace("\n","")
    str = ' '.join(str.split())
    str = str.replace(" ", ",")
    list = literal_eval(str)
    return list

# to turn dataframe with columns that have string values into list values
def df_columns_string_to_list(df,list_of_columns):
    for column in list_of_columns:
        df.loc[:, column] = df.loc[:, column].apply(my_literal_eval)
    return df



##

