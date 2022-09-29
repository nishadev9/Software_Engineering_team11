# This weeks code focuses on understanding basic functions of pandas and numpy
# This will help you complete other lab experiments

import numpy as np
import pandas as pd


def create_numpy_ones_array(shape, dtype=int):

    return np.ones(shape, dtype)


def create_numpy_zeros_array(shape, dtype=int):

    return np.zeros(shape, dtype)


def create_identity_numpy_array(order):
    # if shape
    a = np.zeros((order, order))
    for i in range(order):
        a[i][i] = 1
    return a


def matrix_cofactor(array):
    return np.linalg.inv(array).T * np.linalg.det(array)


def f1(X1, coef1, X2, coef2, seed1, seed2, seed3, shape1, shape2):
    np.random.seed(seed1)
    W1 = np.random.rand(shape1[0], shape1[1])


    np.random.seed(seed2)
    W2 = np.random.rand(shape2[0], shape2[1])

    try:
	    t1 = X1.copy()
	    for _ in range(coef1):
	        t1 = np.matmul(t1, X1)

	    t2 = X2.copy()
	    for _ in range(coef2):
	        t2 = np.matmul(t2, X2)

    except Exception as e:
        return -1

    try:
        y1 = np.matmul(W1, t1)+np.matmul(W2, t2)

    except Exception as e:
        return -1



    np.random.seed(seed3)
    s = y1.shape
    b=np.random.rand(s[0], s[1])
    return y1+b


def fill_with_mode(filename, column):
    """
    Fill the missing values(NaN) in a column with the mode of that column
    Args:
        filename: Name of the CSV file.
        column: Name of the column to fill
    Returns:
        df: Pandas DataFrame object.
        (Representing entire data and where 'column' does not contain NaN values)
        (Filled with above mentioned rules)
    """
    df = pd.read_csv(filename)
    df[column].fillna(df[column].mode()[0], inplace=True)
    return df


def fill_with_group_average(df, group, column):
    """
    Fill the missing values(NaN) in column with the mean value of the 
    group the row belongs to.
    The rows are grouped based on the values of another column

    Args:-
        df: A pandas DataFrame object representing the data.
        group: The column to group the rows with
        column: Name of the column to fill
    Returns:
        df: Pandas DataFrame object.
        (Representing entire data and where 'column' does not contain NaN values)
        (Filled with above mentioned rules)
    """
    df[column].fillna(df.groupby(
        group)[column].transform('mean'), inplace=True)

    return df


def get_rows_greater_than_avg(df, column):
    """
    Return all the rows(with all columns) where the value in a certain 'column'
    is greater than the average value of that column.

    row where row.column > mean(data.column)

    Args:
        df: A pandas DataFrame object representing the data.
        column: Name of the column to fill
    Returns:
        df: Pandas DataFrame object.
    """
    return df[df[column] > df[column].mean()]



# f1(X1, coef1, X2, coef2, seed1, seed2, seed3, shape1, shape2)


print("waddduppppppppppp dawg")
print(f1(np.array([[1,2,3],[4,5,5]]),3,np.array([[7,8],[10,12]]),2,1,2,3,(3,2),(3,2)))
