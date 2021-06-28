import scipy.stats as stats
import numpy as np
from scipy.special import entr
from yellowbrick.model_selection import FeatureImportances

def get_user_object(tweets):
    return tweets[0]['user']

def get_user_id(tweets):
    return tweets[0]['user']['id_str']

def get_user_name(tweets):
    return tweets[0]['user']['name']

def get_user_screen_name(tweets):
    return tweets[0]['user']['screen_name']

def get_retweets(tweets):
    rts=[]
    for t in tweets:
        if 'retweeted_status' in t:
            rts.append(t)
    return rts

def get_only_tweets(tweets):
    ts=[]
    for t in tweets:
        if 'retweeted_status' not in t:
            ts.append(t)
    return ts

def get_all_texts(tweets):
    all_texts=[]
    for t in tweets:
        if 'full_text' in t:
            all_texts.append(t['full_text'])
        else:
            all_texts.append(t['text'])
    return all_texts

def get_statistical_results_of_list(aList):
    if len(aList)>=1:
        return min(aList), max(aList), np.mean(aList), np.median(aList), np.std(aList), \
               stats.skew(aList), stats.kurtosis(aList), entr(np.array(aList)).sum()
    else:
        return 0,0,0,0,0,0,0,0

def covers_needs(tweets):
    rts = get_retweets(tweets)
    ts = get_only_tweets(tweets)
    if len(rts)>=2 and len(ts)>=2:
        # print('arithmos tweets,rts,total, user:', len(ts),len(rts),len(tweets),tweets[0]['user']['id_str'],tweets[0]['dataset'])
        return True
    else:
        return False

def feature_importance(model,X,Y):
    viz = FeatureImportances(model)
    viz.fit(X, Y)
    return [f for f in reversed(viz.features_)]

