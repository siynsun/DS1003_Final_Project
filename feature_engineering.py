#to do: missing indicator, mvp
from sklearn import preprocessing
from scipy import stats
import numpy as np
import pandas as pd
import math
import gpxpy.geo

# check skewness for all numerical variables(including price) and log transform skewed numeric features
def log_skewness(data, num_cols, Y):

    #numeric_feats = data.dtypes[data.dtypes != "object"].index
    #skewed_feats = data[num_cols].apply(lambda x: stats.skew(x))  # compute skewness
    #skewed_feats = skewed_feats[skewed_feats > 0.8]
    #skewed_feats = skewed_feats.index
    #data[skewed_feats] = np.log1p(data[skewed_feats])

    data[Y] = np.log(data[Y]) # log(price)

    return data


# detect outliers of target price and delete outliers
# outliers are price which outside 2 standard deviations(or 1.96) # 2.575
def del_outliers(data, Y='price'):

    outliers = []
    data_array = np.array(data[Y])
    mean = np.mean(data_array)
    std = np.std(data_array)

    for i in range(0, len(data_array)):
        if abs(data_array[i] - mean) > 1.96 * std:
            outliers.append(data_array[i])
    thred = sorted(outliers)[0]
    data[Y][data[Y] > thred] = np.nan
    data = data.dropna()

    return data


# standardization
def standardize_matrix(matrix):

    for col in range(matrix.shape[1]):
        mean = np.mean(matrix[:,col])
        std = np.std(matrix[:,col])
        matrix[:, col] = list(map(lambda x: (x - mean) / std, matrix[:,col]))

    return matrix


def standardize_df(data):

    for col in data.columns.tolist():
        if col != 'price': # not standardize target
            mean = np.mean(data[col])
            std = np.std(data[col] )
            data[col] = data[col].apply(lambda x: (x - mean) / std)

    return data


# normalization
def normalize_df(data):

    for col in data.columns.tolist():
        if col != 'price': # not standardize target
            data_col_max, data_col_min = [data[col].max(axis=0), data[col].min(axis=0)]
            data[col] = data[col].apply(lambda x: x - data_col_min) / (data_col_max - data_col_min)

    return data


# create feature: distance to subway


def prepare_subway_data():
    '''
    data source: https://data.ny.gov/Transportation/NYC-Transit-Subway-Entrance-And-Exit-Data/i9wp-a4ja
    '''
    # read external subway dataset
    subway = pd.read_csv('./data/subway.csv')
    useful_column = ['Station Longitude', 'Station Latitude', 'Station Name']
    subway = subway[useful_column]
    subway = subway.dropna()
    subway = subway.drop_duplicates()
    subway = subway.reset_index(drop=True)

    return (subway)


def prepare_park_data():
    '''
    data source: https://data.cityofnewyork.us/Recreation/NYC-Parks-Public-Events-Upcoming-14-Days/w3wp-dpdi
    '''
    # read external park dataset
    park = pd.read_json('./data/park.json')
    park = pd.DataFrame(park)
    useful_column = ['parknames', 'coordinates']
    park = park[useful_column]

    # clean park dataset
    park = park.drop_duplicates()
    park = park.replace(r'', np.nan, regex=True)
    park = park.dropna()
    park['latitude'], park['longitude'] = park['coordinates'].str.split(', ', 1).str
    park = park.drop('coordinates', 1)
    # park[park['longitude'].str.contains(";")] # row 77, 121, 123 are anomalous
    park.drop(park.index[[77, 121, 123]], inplace=True)
    park = park.reset_index(drop=True)
    park['latitude'] = park['latitude'].astype('float64')
    park['longitude'] = park['longitude'].astype('float64')

    return (park)



def create_new_feature(data):
    '''
    subway feature created: count_near_subway, dist_to_nearest_subway
    park feature created: count_near_park, dist_to_nearest_park
    '''

    park = prepare_park_data()
    subway = prepare_subway_data()

    data = data.reset_index(drop=True)

    # create two subway features
    data['count_near_subway'] = np.zeros(len(data))
    data['dist_to_nearest_subway'] = np.zeros(len(data))
    data['dist_to_nearest_subway'] = 1600

    # create two park features
    data['count_near_park'] = np.zeros(len(data))
    data['dist_to_nearest_park'] = np.zeros(len(data))
    data['dist_to_nearest_park'] = 1600

    # set up threshold, point to point distance, unit: meters (i.e. 0.25 miles)
    dist_to_subway = 400
    dist_to_park = 800

    for i in range(data.shape[0]):
        # house location
        lat1 = data['latitude'][i]
        lon1 = data['longitude'][i]
        min_dist12 = 1600
        min_dist13 = 1600
        print (i)
        for j in range(subway.shape[0]):
            # subway location
            lat2 = subway['Station Latitude'][j]
            lon2 = subway['Station Longitude'][j]

            # args order is super important
            dist12 = gpxpy.geo.haversine_distance(lat1, lon1, lat2, lon2)

            if dist12 <= dist_to_subway:
                data['count_near_subway'][i] += 1

            if dist12 < min_dist12:
                data['dist_to_nearest_subway'][i] = dist12
                min_dist12 = dist12

            try:
                lat3 = park['latitude'][j]
                lon3 = park['longitude'][j]
                dist13 = gpxpy.geo.haversine_distance(lat1, lon1, lat3, lon3)
                if dist13 <= dist_to_park:
                    data['count_near_park'][i] += 1

                if dist13 < min_dist13:
                    data['dist_to_nearest_park'][i] = dist13
                    min_dist13 = dist13
            except KeyError:
                continue

    return (data)


'''
datapath = './data/encoded_others.pkl'
pkl_file = open(datapath, 'rb')
dataset = pickle.load(pkl_file)
'''

