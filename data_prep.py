import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
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
    df = df[binary_cols +  cat_cols + text_cols + num_cols + mix_cols + Y]

    # drop obs with larger than 9 features missing
    df = df[df.isnull().sum(axis=1) <= 9]
    return df

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

# category_counter for mixed type columns 
def cat_counter(df, col):
    cat_list = []
    for k, row in df.iterrows():
        row[col] = row[col].strip("{}[]")
        row[col] = row[col].replace(" '", "")
        row[col] = row[col].replace("'", "")
        row[col] = row[col].replace('"', "")
        for j in row[col].split(','):
            if j not in cat_list and j not in ['', 'None']:
                cat_list.append(j)
        df.iloc[k, df.columns.get_loc(col)] = row[col]
    return cat_list

# build features
def extract_features(feature_df, binary_cols, cat_cols, text_cols, num_cols, mix_cols):
    # Binary: Drop the row if NA
    for binary_col in binary_cols:
        feature_df = feature_df[pd.notnull(feature_df[binary_col])]
    for binary_col in binary_cols:
        feature_df[binary_col] = feature_df[binary_col].map({'f': 0, 't': 1})

    # Categorical: One-hot
    feature_df = pd.get_dummies(feature_df, columns=cat_cols)
    
    # TODO: extract text feature
    for text_col in text_cols:
        tfidf_vectorizer = CountVectorizer(max_df=0.9, min_df=3, stop_words='english')
        tfidf = tfidf_vectorizer.fit_transform(feature_df[text_col])
        lda = LatentDirichletAllocation(n_topics=10, learning_method='online',
                                        random_state=0)
        lda_res = lda.fit_transform(tfidf)
        topic = text_col + '_topic'
        feature_df[topic] = np.argmax(lda_res, axis=1)
        feature_df = feature_df.drop([text_col], 1)
        tfidf_feature_names = tfidf_vectorizer.get_feature_names()
        print_top_words(lda, tfidf_feature_names, 10)
        
    # Mixed: One-hot
    feature_df = feature_df.reset_index(drop=True)
    for mix_col in mix_cols:
        mix_list = cat_counter(feature_df, mix_col)
        for mix in mix_list:
            feature_df[mix] = 0
        for k, row in feature_df.iterrows():
            for j in row[mix_col].split(','):
                if j not in ['', 'None']:
                    feature_df.iloc[k, feature_df.columns.get_loc(j)] = 1
        feature_df = feature_df.drop([mix_col], 1)
                                    
    return feature_df

if __name__ == '__main__':
    # change the direction to where your data located
    os.chdir('/Users/Sean/Desktop/DS1003_Final_Project')
    sys.path.append('/Users/Sean/Desktop/DS1003_Final_Project')

    # select meaningful features
    binary_cols = ['host_is_superhost', 'instant_bookable']
    cat_cols = ['host_response_time', 'zipcode', 'property_type', 'room_type',
                'bed_type', 'cancellation_policy']
    text_cols = ['name', 'summary', 'space', 'description', 'neighborhood_overview',
                 'transit', 'access', 'interaction', 'house_rules', 'host_about']
    num_cols = ['host_response_rate', 'host_listings_count',
                'accommodates', 'bathrooms', 'bedrooms', 'beds', 'guests_included',
                'minimum_nights', 'maximum_nights', 'calculated_host_listings_count', 'availability_30','availability_365']
    mix_cols = ['host_verifications', 'amenities']
    Y = ['price']

    # read
    clean_data = read_data('./data/listings_all.csv')
    clean_data = clean_data.head(30000)

    # missing imputation
    mi.text_imputation(clean_data, text_cols)
    mi.mean_imputation(clean_data, num_cols)

    # feature engineering
    #to do: missing indicator, mvp, .., create new features

    # encoding
    encoded_df = extract_features(clean_data,binary_cols,cat_cols,text_cols,num_cols,mix_cols)

    # out
    out = open('./data/encoded_df.pkl', 'wb')
    pickle.dump(encoded_df, out)
    out.close()