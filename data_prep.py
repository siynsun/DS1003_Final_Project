import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import LatentDirichletAllocation
import missing_imputation as mi
import os
import sys


# read the data
def read_data(file_path):
    df = pd.read_csv(file_path, encoding = "ISO-8859-1")

    # drop all the columns having missing greater than 50%
    na_cols = df.columns[pd.isnull(df).sum() / len(df) > 0.5].tolist()
    df.drop(na_cols, inplace=True, axis=1)

    # only keep selected variables
    df = df[text_cols + cat_cols + num_cols + Y]

    # drop obs with larger than 9 features missing
    df = df[df.isnull().sum(axis=1) <= 9]
    return df


# build features
def extract_features(feature_df, cat_cols, text_cols, num_cols):
    # Encode text features
    for text_col in text_cols:
        tfidf_vec = TfidfVectorizer(stop_words="english", max_df=80, min_df=5, ngram_range=[1, 1])
        lda = LatentDirichletAllocation()
        tfidf_tokens = tfidf_vec.fit_transform(feature_df[text_col])
        lda_res = lda.fit_transform(tfidf_tokens)
        topics = text_col + 'topic'
        feature_df[topics] = np.argmax(lda_res, axis=1)
        feature_df = feature_df.drop([text_col], 1)

    # Encode categorical features
    for cat_col in cat_cols:
        all_unique_val = np.unique(feature_df[cat_col])
        for val in all_unique_val:
            feature_df["{0}={1}".format(cat_col, val)] = feature_df.apply(lambda x: x[cat_col] == val, 1)
        feature_df = feature_df.drop(cat_col, 1)

    # Encode Label
    label_encoder = LabelEncoder()
    label_encoder.fit(feature_df['price'])
    feature_df['Y'] = label_encoder.transform(feature_df['price'])
    encoded_df = feature_df.drop(['price'], 1)

    return encoded_df


if __name__ == '__main__':
    # change the direction to where your data located
    os.chdir('/Users/Sean/Desktop/DS1003_Final_Project')
    sys.path.append('/Users/Sean/Desktop/DS1003_Final_Project')

    # select meaningful features
    cat_cols = ['host_response_time', 'host_is_superhost', 'host_has_profile_pic', 'host_identity_verified',
                'neighbourhood_group_cleansed',
                'zipcode', 'property_type', 'room_type', 'bed_type', 'instant_bookable']
    text_cols = ['summary', 'name', 'space', 'description', 'neighborhood_overview', 'transit', 'access', 'interaction',
                 'house_rules',
                 'host_about', 'host_verifications', 'neighbourhood_cleansed', 'amenities']
    num_cols = ['host_response_rate', 'host_listings_count', 'host_total_listings_count', 'accommodates', 'bathrooms',
                'bedrooms',
                'beds', 'guests_included', 'minimum_nights', 'number_of_reviews', 'review_scores_rating',
                'reviews_per_month']
    Y = ['price']

    # read
    clean_data = read_data('./data/listings_all.csv')

    # missing imputation
    mi.mean_imputation(clean_data, num_cols)
    mi.text_imputation(clean_data, text_cols)
    mi.mode_imputation(clean_data, cat_cols)
    missing_list = mi.validate_missing(clean_data)
    print(missing_list)

    # feature engineering
    #to do: missing indicator, mvp, .., create new features

    # encoding
    encoded_df = extract_features(clean_data,cat_cols,text_cols,num_cols)

    # out
    out = open('./data/encoded_df.pkl', 'wb')
    pickle.dump(encoded_df, out)
    out.close()