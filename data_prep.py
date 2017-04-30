import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob
import missing_imputation as mi
import feature_engineering as fg
from importlib import reload
#reload(fg)
import os
import sys

# read the data
def read_data(file_path):
    df = pd.read_csv(file_path, encoding = "ISO-8859-1")

    # drop all the columns having missing greater than 50%
    na_cols = df.columns[pd.isnull(df).sum() / len(df) > 0.5].tolist()
    df.drop(na_cols, inplace=True, axis=1)

    # only keep selected variables (32 features)
    df = df[binary_cols +  cat_cols + text_cols + num_cols + mix_cols + Y]

    # drop obs with larger than 8 features missing (25%)
    df = df[df.isnull().sum(axis=1) < 9]
    return df

# print topic top words for text features
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
    # Binary: 
    #for binary_col in binary_cols: # Drop the row if NA
    #    feature_df = feature_df[pd.notnull(feature_df[binary_col])]
    for binary_col in binary_cols: # Encode 0/1
        feature_df[binary_col] = feature_df[binary_col].map({'f': 0, 't': 1})

    # Categorical: One-hot
    feature_df = pd.get_dummies(feature_df, columns=cat_cols)
    
    # Text:
    for text_col in text_cols: # Sentiment 
        feature_df[text_col + '_polarity'] = feature_df.apply(lambda x: 
            TextBlob(x[text_col]).sentiment.polarity, axis=1)
        feature_df[text_col + '_subjectivity'] = feature_df.apply(lambda x: 
            TextBlob(x[text_col]).sentiment.subjectivity, axis=1)
    for text_col in text_cols: # Count + LDA
        count_vectorizer = CountVectorizer(max_df=0.9, min_df=3, stop_words='english')
        count = count_vectorizer.fit_transform(feature_df[text_col])
        lda = LatentDirichletAllocation(n_topics=10, learning_method='online',
                                        random_state=0)
        lda_res = lda.fit_transform(count)
        feature_df[text_col + '_topic'] = np.argmax(lda_res, axis=1)
        feature_df = feature_df.drop([text_col], 1)
        tfidf_feature_names = count_vectorizer.get_feature_names()
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
    ### change the direction to where your data located
    #os.chdir('/Users/Sean/Desktop/DS1003_Final_Project')
    #sys.path.append('/Users/Sean/Desktop/DS1003_Final_Project')

### select meaningful features
    binary_cols = ['host_is_superhost', 'instant_bookable']
    cat_cols = ['host_response_time', 'zipcode', 'property_type', 'room_type',
                'bed_type', 'cancellation_policy']
    text_cols = ['name', 'summary', 'space', 'description', 'neighborhood_overview',
                 'transit', 'access', 'interaction', 'house_rules', 'host_about']
    num_cols = ['host_response_rate', 'host_listings_count', 'extra_people',
                'accommodates', 'bathrooms', 'bedrooms', 'beds', 'guests_included',
                'minimum_nights', 'maximum_nights', 'calculated_host_listings_count', 'longitude', 'latitude']
    mix_cols = ['host_verifications', 'amenities']
    Y = ['price']

    ### read data
    clean_data = read_data('./data/listings_all.csv')
    #clean_data = clean_data.head(3000)

    ### missing imputation

    # text imputation feature: transit, summary, access, house_rules, host_about, interaction, neighborhood_overview, description, space
    mi.text_imputation(clean_data, text_cols)

    # mode imputation feature: 'host_is_superhost', 'instant_bookable'
    mi.mode_imputation(clean_data, binary_cols)

    # knn imputation feature: 'host_is_superhost', 'host_response_time', 'zipcode', 'host_response_rate', 'host_listings_count', 'bathrooms', 'bedrooms', 'beds'
    clean_data = mi.knn_imputation(clean_data, 'zipcode', 3, cat_cols, num_cols)
    clean_data = mi.knn_imputation(clean_data, 'host_response_time', 3, cat_cols, num_cols)
    clean_data = mi.knn_imputation(clean_data, 'host_response_rate', 3, cat_cols, num_cols)
    clean_data = mi.knn_imputation(clean_data, 'host_listings_count', 3, cat_cols, num_cols)

    # replace 0 count of bathrooms, bedrooms, beds to np.nan
    clean_data['bathrooms'].replace(0, np.nan)
    clean_data['bedrooms'].replace(0, np.nan)
    clean_data['beds'].replace(0, np.nan)
    clean_data = mi.knn_imputation(clean_data, 'bathrooms', 3, cat_cols, num_cols)
    clean_data = mi.knn_imputation(clean_data, 'bedrooms', 3, cat_cols, num_cols)
    clean_data = mi.knn_imputation(clean_data, 'beds', 3, cat_cols, num_cols)

    # validate
    if clean_data.columns[pd.isnull(clean_data).sum() / len(clean_data) > 0].tolist() == []:
        print ("**All missing values have been filled**")

    ### feature engineering

    # deal with outliers
    clean_data = fg.del_outliers(clean_data)
    print("**Outliers have been deleted**")

    # deal with skewness
    clean_data = fg.log_skewness(clean_data, num_cols, Y)
    print("**Skewed features have been transformed**")

    # create new  features
    clean_data = fg.create_new_feature(clean_data)
    print("**New features have been created**")

    ### encoding
    encoded_df = extract_features(clean_data,binary_cols,cat_cols,text_cols,num_cols,mix_cols)
    print("**Data has been encoded**")

    # standardize and normalize data
    encoded_df = fg.standardize_df(encoded_df)
    encoded_df = fg.normalize_df(encoded_df)
    print("**Data has been standardized**")

    # drop longitude and latitude
    encoded_df = encoded_df.drop('longitude', 1)
    encoded_df = encoded_df.drop('latitude', 1)

    # divide dataset to two part by room_type
    encoded_entire = encoded_df[encoded_df['room_type_Entire home/apt'] == 1]
    encoded_entire = encoded_entire.drop('room_type_Entire home/apt', 1)
    encoded_entire = encoded_entire.drop('room_type_Private room', 1)
    encoded_entire = encoded_entire.drop('room_type_Shared room', 1)

    encoded_others = encoded_df[encoded_df['room_type_Entire home/apt'] != 1]
    encoded_others = encoded_others.drop('room_type_Entire home/apt', 1)
    encoded_others = encoded_others.drop('room_type_Shared room', 1)


    ### output the cleaned, encoded dataset for modeling
    out1 = open('./data/encoded_entire.pkl', 'wb')
    pickle.dump(encoded_entire, out1)

    out2 = open('./data/encoded_others.pkl', 'wb')
    pickle.dump(encoded_others, out2)

    out1.close()
    out2.close()






