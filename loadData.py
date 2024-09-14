import glob
from sklearn.preprocessing import RobustScaler, StandardScaler, QuantileTransformer, MinMaxScaler
import pandas as pd
from nilearn.connectome import ConnectivityMeasure
import os
import numpy as np


def Get_Upper_Triangle(data):
    x = []
    for i in range(data.shape[0]):
        for j in range(i):
            x.append(data[i, j])
    x = np.array(x)
    return x


def Read_1D(filename):
    roi_signal = pd.read_csv(filename, sep='\t').values

    return roi_signal


def Load_Atlas_Data(dirpath, extension, metadata_df, norm="roboust", mode='abide'):

    X = []
    Y = []
    Age = []
    Sex = []
    Site = []

    for file in glob.glob(dirpath + '/*.' + extension):

        basename = os.path.basename(file)
        # CMU_a_0050649_rois_aal
        if mode == 'abide':
            filename = basename.split("_rois")[0]
        else:
            filename = basename.split("_func_preproc")[0]
        # print(filesitename)
        related_row = metadata_df.loc[metadata_df['FILE_ID'] == filename]
        if related_row.shape[0] == 0 or related_row.shape[0] > 1:
            print(basename)
            breakpoint()
        else:
            age = related_row['Age'].values[0]
            classlabel = related_row['Group'].values[0]
            sex = related_row['Sex'].values[0]
            siteid = related_row['SITE_ID'].values[0]

            if str(age) != 'nan':
                if mode == 'abide':
                    time_series = Read_1D(file)
                    correlation_measure = ConnectivityMeasure(kind='correlation')
                    correlation_matrix = correlation_measure.fit_transform([time_series])[0]
                else:
                    correlation_matrix = np.load(file)

                correlation_matrix = Get_Upper_Triangle(correlation_matrix)
                X.append(correlation_matrix)

                if classlabel == "Autism":
                    Y.append(1)
                elif classlabel == 'Control':
                    Y.append(0)
                else:
                    print(classlabel)
                    breakpoint()

                Age.append(age)
                Sex.append(sex)
                Site.append(siteid)

    X = np.array(X)
    Age = np.array(Age)
    Sex = np.array(Sex)
    Site = np.array(Site)

    # Do some useful preprocess such as constnat removal, correlated features removal and low variance feature removal

    #  print("Main Data Size : {} . {}".format(X.shape[0], X.shape[1]))

    Y = np.array(Y)

    if norm == "stand":
        X = StandardScaler().fit_transform(X)
    elif norm == "roboust":
        X = RobustScaler().fit_transform(X)
    elif norm == "minmax":
        X = MinMaxScaler().fit_transform(X)

    batch_data = pd.DataFrame({"batch": (Site), "gender": (Sex), "age": (Age)})

    return X, Y, batch_data
