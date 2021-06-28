import pickle
from features import feature_tools, feature_extractor
from tqdm import tqdm
from pymongo import MongoClient

conn = MongoClient('mongodb://...')
db = conn['#DB_NAME']
col=db['#COLLECTION_NAME']

datasets = ['Traditional_Spambots_3', 'twittertechnology', 'Verified', 'Pron', 'Gilani', 'Political',  'News_Agency',
            'Astroturf_political', 'thefakeproject', 'Traditional_Spambots_1', 'Company', 'botwiki', 'Social_Spambots_3', 'Celebrities', 'Cresci_RTbust',
            'Botometer', 'Vendor', 'Varol',  'Traditional_Spambots_4', 'Social_Spambots_2', 'intertwitter',
            'italian_elections', 'fastfollowerz', 'Social_Spambots_1','Genuine_Accounts', 'Cresci_Stock','Midterm','Caverlee']


def filterTheDict(dictObj, callback):
    newDict = dict()
    # Iterate over all the items in dictionary
    for (key, value) in dictObj.items():
        # Check if item satisfies the given condition then add to new dict
        if callback((key, value)):
            newDict[key] = value
    return newDict

def create_full_training_test_set(datasets):
    print ('...Full feature extraction...')
    ###load user labels: format Dict {"user_id":String , "label": "bot" or "human" or "cyborg"}
    labels = pickle.load(open('user_labels', 'rb'))
    final_data = []
    ###User datasets dict -> see Available Users.csv
    user_datasets_dict = pickle.load(open('pickles/user_dataset_mapping', 'rb'))
    for d in tqdm(datasets):
        collection=db[d]
        dataset_users = filterTheDict(user_datasets_dict, lambda elem: elem[1] == d)
        print (d,len(dataset_users.keys()))
        for u in tqdm(dataset_users.keys()):
            utweets = []
            iterator = collection.find({'user.id_str': u},{'id':0,'id_str':0,'truncated':0,'display_text_range':0,'geo':0,'coordinates':0,'contributors':0,'protected':0,'utc_offset':0,'time_zone':0,'geo_enabled':0,'contributors_enabled':0,'is_translator':0,'profile_background_color':0,"profile_background_color" : 0,"profile_background_image_url" : 0,
        "profile_background_image_url_https" : 0, "profile_background_tile" : 0, "profile_link_color" : 0,"profile_sidebar_border_color" : 0, "profile_sidebar_fill_color" : 0,
        "profile_text_color" : 0, "profile_use_background_image" : 0, "profile_image_url" : 0,"profile_image_url_https" : 0, "following" : 0,
        "follow_request_sent" : 0, "notifications" : 0, "geo" : 0,  "coordinates" : 0, "contributors":0})
            for i in iterator:
                utweets.append(i)
            if feature_tools.covers_needs(utweets) == True:
                user_dict = utweets[0]['user']
                features = feature_extractor.get_all_features_train(user_dict, utweets, labels)
                final_data.append(features)
            else:
                continue
    pickle.dump(final_data, open('final_data_v2', 'wb'))

def create_full_training_test_set_no_rts(datasets):
     ###load user labels: format Dict {"user_id":String , "label": "bot" or "human" or "cyborg"}
    labels = pickle.load(open('user_labels', 'rb'))
    final_data = []
    ###User datasets dict -> see Available Users.csv
    user_datasets_dict = pickle.load(open('pickles/user_dataset_mapping', 'rb'))
    for d in tqdm(datasets):
        print('...Pruned feature extraction...')
        collection = db[d]
        dataset_users = filterTheDict(user_datasets_dict, lambda elem: elem[1] == d)
        print(d, len(dataset_users.keys()))
        for u in tqdm(dataset_users.keys()):
            utweets = []
            iterator = collection.find({'user.id_str': u},
                                       {'id': 0, 'id_str': 0, 'truncated': 0, 'display_text_range': 0, 'geo': 0,
                                        'coordinates': 0, 'contributors': 0, 'protected': 0, 'utc_offset': 0,
                                        'time_zone': 0, 'geo_enabled': 0, 'contributors_enabled': 0, 'is_translator': 0,
                                        'profile_background_color': 0, "profile_background_color": 0,
                                        "profile_background_image_url": 0,
                                        "profile_background_image_url_https": 0, "profile_background_tile": 0,
                                        "profile_link_color": 0, "profile_sidebar_border_color": 0,
                                        "profile_sidebar_fill_color": 0,
                                        "profile_text_color": 0, "profile_use_background_image": 0,
                                        "profile_image_url": 0, "profile_image_url_https": 0, "following": 0,
                                        "follow_request_sent": 0, "notifications": 0, "geo": 0, "coordinates": 0,
                                        "contributors": 0})
            for i in iterator:
                utweets.append(i)
            if len(utweets)>=2:
                user_dict = utweets[0]['user']
                try:
                    features = feature_extractor.get_all_features_no_rts_train(user_dict, utweets, labels)
                    final_data.append(features)
                except:
                    print ('error')
            else:
                continue
    pickle.dump(final_data, open('final_data_no_rts_v2', 'wb'))

