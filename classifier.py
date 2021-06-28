import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from collections import Counter
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, AdaBoostClassifier
from sklearn import metrics
from optional import return_user_specific_class
from sklearn.model_selection import GridSearchCV,train_test_split


user_feats = pickle.load(open('features/user_features','rb'))
content_feats = pickle.load(open('features/content_features','rb'))
temporal_fts_all = pickle.load(open('features/temporal_features','rb'))
sentiment_feats = pickle.load(open('features/sentiment_features','rb'))
friend_feats = pickle.load(open('features/friend_features','rb'))
temporal_fts_twts = pickle.load(open('features/temporal_features_twts','rb'))
network_fts = pickle.load(open('features/network_features','rb'))
content_fts_twts = pickle.load(open('features/content_features_twts','rb'))
# feature_set = {1:user_feats,2:content_feats,3:temporal_fts_all,4:sentiment_feats,5:friend_feats,6:network_fts}
feature_list_only_tweets = {1:user_feats,2:content_fts_twts,3:temporal_fts_twts,4:sentiment_feats,5:network_fts}
pickle.dump(feature_list_only_tweets,open('features_short','wb'))

def filterTheDict(dictObj, callback):
    newDict = dict()
    # Iterate over all the items in dictionary
    for (key, value) in dictObj.items():
        # Check if item satisfies the given condition then add to new dict
        if callback((key, value)):
            newDict[key] = value
    return newDict

def create_gridsearch_model():
    df = pd.DataFrame(pickle.load(open('final_data_v2', 'rb')))
    crescis = []
    usersdata = pickle.load(open('pickles/user_dataset_mapping', 'rb'))
    for k, v in usersdata.items():
        if v == 'Cresci_Stock':
            crescis.append(k)
    df=df[df['user_id'].isin(crescis)]
    if 'max_appearance_of_punc_mark' in df.columns:
        df = df.drop(['max_appearance_of_punc_mark'], axis=1)
    print(df.head())
    print(Counter(df.columns))
    df['label'] = df['label'].map({'human': 0, 'bot': 1, 'cyborg': 1})
    print(df.isna().sum().sum())
    print(df['label'].value_counts())
    X = df.drop(['user_name', 'user_screen_name', 'user_id', 'label','time'], axis=1)
    print(X.shape)
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    model_params = {
        'max_depth': [None,1,5,15,20],
        # 'max_features': [ 0.33 , 1.0],
        # 'min_samples_leaf': [1,5,10,15],
        'n_estimators': list(range(150,180,10))
    }
    rf_model = RandomForestClassifier(random_state=1)
    clf = GridSearchCV(rf_model, model_params,verbose=100,cv=3)
    model = clf.fit(X, y)
    from pprint import pprint
    pprint(model.best_estimator_.get_params())

def one_model(datatype):
    if datatype=='full':
        df = pd.DataFrame(pickle.load(open('final_data_v2', 'rb')))
    else:
        datatype='short'
        df = pd.DataFrame(pickle.load(open('final_data_no_rts_v2', 'rb')))
    if 'max_appearance_of_punc_mark' in df.columns:
        df = df.drop(['max_appearance_of_punc_mark'], axis=1)
    df['label'] = df['label'].map({'human': 0, 'bot': 1,'cyborg':1})
    print ('Number of none values:',df.isna().sum().sum())
    print('Number of labelled:'+'\n', df['label'].value_counts())
    X = df.drop(['user_name', 'user_screen_name', 'user_id', 'label', 'time'], axis=1)
    print ('X shape',X.shape)
    y = df['label']
    sm = ADASYN(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=10)
    X_train,y_train = sm.fit_resample(X_train,y_train)
    print(y_train.value_counts())
    model = RandomForestClassifier(random_state=14, n_jobs=-1, n_estimators=100, max_depth=10, min_samples_leaf=2)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("F1 score:", metrics.f1_score(y_test, y_pred))
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred, target_names=['human', 'bot']))
    pickle.dump(model, open('binary_'+datatype, 'wb'))

def create_binary_specific_class(botclass,datatype):
    if datatype == 'full':
        df = pd.DataFrame(pickle.load(open('final_data_v2', 'rb')))
    else:
        datatype = 'short'
        df = pd.DataFrame(pickle.load(open('final_data_no_rts_v2', 'rb')))
    from optional import return_user_specific_class
    multi = return_user_specific_class()
    df['label'] = df['user_id'].map(multi)
    df=df[df['label'].isin(['human',botclass])]
    if 'max_appearance_of_punc_mark' in df.columns:
        df = df.drop(['max_appearance_of_punc_mark'], axis=1)
    df['label'] = df['label'].map(
        {'human': 0, botclass: 1})
    X = df.drop(['user_name', 'user_screen_name', 'user_id', 'label','time'], axis=1)
    y = df['label']
    sm = ADASYN(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=10)
    print (y_train.value_counts())
    X_train, y_train = sm.fit_resample(X_train, y_train)
    rf = RandomForestClassifier(random_state=14, n_jobs=-1, n_estimators=170,max_depth=10)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    pickle.dump(X_test,open(botclass+'_'+datatype+'_Xtest','wb'))
    pickle.dump(y_test, open(botclass+'_'+datatype + '_ytest', 'wb'))
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred,target_names=['human', botclass]))
    pickle.dump(rf,open('models_balanced_train/'+botclass+'_'+datatype,'wb'))

def listToString(s):
    str1 = " "
    return (str1.join(s))

def create_one_vs_rest_classifiers(datatype):
    print (datatype)
    from sklearn.multiclass import OneVsRestClassifier
    if datatype == 'full':
        df = pd.DataFrame(pickle.load(open('final_data_v2', 'rb')))
    else:
        datatype = 'short'
        df = pd.DataFrame(pickle.load(open('final_data_no_rts_v2', 'rb')))
    from optional import return_user_specific_class
    multi = return_user_specific_class()
    df['label'] = df['user_id'].map(multi)
    df = df[~df['label'].isin(['otherbots'])]
    print (df.head(2))
    if 'max_appearance_of_punc_mark' in df.columns:
        df = df.drop(['max_appearance_of_punc_mark'], axis=1)
    df['label'] = df['label'].map({'human':0,'socialbot':1, 'politicalbot':2,  'spambot':3, 'selfdeclaredbots':4, 'cyborg':5})
    print('Number of none values:', df.isna().sum().sum())
    print('Number of labelled:' + '\n', df['label'].value_counts())
    X = df.drop(['user_name', 'user_screen_name', 'user_id', 'label','time'], axis=1)
    y = df['label']
    sm = ADASYN(random_state=42)
    print(y.value_counts())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)
    X_train, y_train = sm.fit_sample(X_train, y_train)
    rf = RandomForestClassifier(random_state=14, n_jobs=-1, n_estimators=170, max_depth=10)
    onevsrest = OneVsRestClassifier(rf)
    print ('...fitting...')
    onevsrest.fit(X_train,y_train)
    pickle.dump(onevsrest, open('onevsrest_'+datatype, 'wb'))
    y_pred = onevsrest.predict(X_test)
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred,target_names=['human', 'socialbot', 'politicalbot', 'spambot', 'selfdeclaredbots', 'cyborg']))

def evaluate_model(X, y, model):
    from sklearn.model_selection import cross_validate
    # cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1)
    # acc = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=6)
    # prec  = cross_val_score(model, X, y, scoring='precision_macro', cv=cv, n_jobs=6)
    # rec = cross_val_score(model, X, y, scoring='recall_macro', cv=cv, n_jobs=6)
    scoring = ['precision_macro','recall_macro']
    scores = cross_validate(model, X, y, cv=2,scoring=scoring, return_train_score=False, n_jobs=6)
    return scores

def direct_multiclass(datatype):
    from sklearn.multiclass import OneVsOneClassifier
    print(datatype)
    if datatype == 'full':
        df = pd.DataFrame(pickle.load(open('data/feature_data/final_data_v2', 'rb')))
    else:
        datatype = 'short'
        df = pd.DataFrame(pickle.load(open('data/feature_data/final_data_no_rts_v2', 'rb')))
    multi = return_user_specific_class()
    df['label'] = df['user_id'].map(multi)
    print(df.head(2))
    if 'max_appearance_of_punc_mark' in df.columns:
        df = df.drop(['max_appearance_of_punc_mark'], axis=1)
    ###only for "without otherbots"###
    df=df[df['label'].isin(['socialbot','politicalbot','human','spambot','selfdeclaredbots','cyborg'])]
    # df['label'] = df['label'].map({'human':0,'socialbot': 1, 'politicalbot': 2, 'spambot': 3, 'selfdeclaredbots': 4, 'otherbots': 5,'cyborg': 6})
    df['label'] = df['label'].map({'socialbot': 1, 'politicalbot': 2, 'human': 0, 'spambot': 3, 'selfdeclaredbots': 4, 'cyborg': 5})
    print('Number of none values:', df.isna().sum().sum())
    print('Number of labelled:' + '\n', df['label'].value_counts())
    X = df.drop(['user_name', 'user_screen_name', 'user_id', 'label','time'], axis=1)
    y = df['label']
    sm = ADASYN(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)
    X_train, y_train = sm.fit_sample(X_train, y_train)
    print (y_train.value_counts)
    rf = RandomForestClassifier(random_state=14, n_jobs=-1, n_estimators=160, max_depth=5)
    rf.fit(X_train,y_train)
    y_pred = rf.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    from sklearn.metrics import classification_report

    print(classification_report(y_test, y_pred,target_names=['human', 'socialbot', 'politicalbot', 'spambot', 'selfdeclaredbots', 'cyborg']))
    pickle.dump(rf,open('direct_multiclass_'+datatype,'wb'))

def create_all_possible_models(datatype):
    results = pd.DataFrame(columns=['Features','Num_of_features','Accuracy', 'F1','Feature_types','Precision','Recall'])
    models={}
    if datatype == 'full':
        df = pd.DataFrame(pickle.load(open('data/feature_data/final_data_v2', 'rb')))
        feature_set = {1: user_feats, 2: content_feats, 3: temporal_fts_all, 4: sentiment_feats, 5: friend_feats,6: network_fts}
    else:
        datatype = 'short'
        df = pd.DataFrame(pickle.load(open('data/feature_data/final_data_no_rts_v2', 'rb')))
        feature_set = {1: user_feats, 2: content_fts_twts, 3: temporal_fts_twts, 4: sentiment_feats,5: network_fts}
    if 'max_appearance_of_punc_mark' in df.columns:
        df = df.drop(['max_appearance_of_punc_mark'], axis=1)
    df['label'] = df['label'].map({'human': 0, 'bot': 1, 'cyborg': 1})
    print(df['label'].value_counts())
    X = df.drop(['user_name', 'user_screen_name', 'user_id', 'label','time'], axis=1)
    print(X.shape)
    y = df['label']
    ada = ADASYN(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)
    print (y_train.value_counts())
    X_train,y_train = ada.fit_resample(X_train,y_train)
    print (y_train.value_counts())
    for L in range(1, len(feature_set.keys())+1):
        for combo in tqdm(combinations(feature_set.keys(), L)):
            print (combo)
            if len(combo)>1:
                data=[]
                data_label =[]
                for i in range(0,len(combo)):
                    data = data + feature_set[combo[i]]
                    data_label.append(combo[i])
            else:
                data = feature_set[combo[0]]
                data_label = combo[0]
            if 'max_appearance_of_punc_mark' in data:
                ind = data.index('max_appearance_of_punc_mark')
                data.pop(ind)
            X_train_i = X_train[data]
            X_test_i = X_test[data]
            model = RandomForestClassifier(random_state=14, n_jobs=-1, n_estimators=170, max_depth=10, min_samples_leaf=2)
            model.fit(X_train_i, y_train)
            y_pred = model.predict(X_test_i)
            print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
            print("F1 score:", metrics.f1_score(y_test, y_pred))
            from sklearn.metrics import classification_report
            print(classification_report(y_test, y_pred, target_names=['human', 'bot']))
            dataset_label = listToString(str(data_label))
            models[dataset_label]=model
            results = results.append({'Features':data_label,'Feature_types':len(combo),'Num_of_features':len(X.columns), 'Accuracy':metrics.accuracy_score(y_test, y_pred),
                                      'F1':metrics.f1_score(y_test, y_pred),'Precision':metrics.precision_score(y_test, y_pred),'Recall':metrics.recall_score(y_test, y_pred)},ignore_index = True)
    pickle.dump(results,open('all_results_'+datatype,'wb'))
    pickle.dump(models, open('all_models_'+datatype, 'wb'))

def stacking_of_binary_classifiers(datatype):
    if datatype == 'full':
        df = pd.DataFrame(pickle.load(open('final_data_v2', 'rb')))
    else:
        datatype = 'short'
        df = pd.DataFrame(pickle.load(open('final_data_no_rts_v2', 'rb')))
    spambot = pickle.load(open('spambot_'+datatype, 'rb'))
    socialbot = pickle.load(open('socialbot_'+datatype, 'rb'))
    politicalbot = pickle.load(open('politicalbot_'+datatype, 'rb'))
    cyborg = pickle.load(open('cyborg_'+datatype, 'rb'))
    selfbots = pickle.load(open('selfdeclaredbots_'+datatype, 'rb'))
    allmodels = [('spambot',spambot),('socialbot',socialbot),('politicalbot',politicalbot),
                 ('cyborg',cyborg),('selfbots',selfbots)]
    rf = RandomForestClassifier(n_estimators=170)
    model = StackingClassifier(estimators=allmodels,final_estimator=rf,cv=5)
    multi = return_user_specific_class()
    df['label'] = df['user_id'].map(multi)
    df = df[~df['label'].isin(['otherbots'])]
    if 'max_appearance_of_punc_mark' in df.columns:
        df = df.drop(['max_appearance_of_punc_mark'], axis=1)
    df['label'] = df['label'].map(
        {'socialbot': 1, 'politicalbot': 2, 'human': 0, 'spambot': 3, 'selfdeclaredbots': 4, 'cyborg': 5})
    X = df.drop(['user_name', 'user_screen_name', 'user_id', 'label','time'], axis=1)
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred,target_names=['human', 'socialbot', 'politicalbot', 'spambot', 'selfdeclaredbots', 'cyborg']))
    pickle.dump(model,open('stacking_model_'+datatype,'wb'))

def ensemble_classifier(datatype):
    if datatype == 'full':
        df = pd.DataFrame(pickle.load(open('data/feature_data/final_data_v2', 'rb')))
    else:
        datatype = 'short'
        df = pd.DataFrame(pickle.load(open('data/feature_data/final_data_no_rts_v2', 'rb')))
    socialbot = pickle.load(open('models_balanced_train/socialbot_'+datatype, 'rb'))
    politicalbot = pickle.load(open('models_balanced_train/politicalbot_'+datatype, 'rb'))
    spambot = pickle.load(open('models_balanced_train/spambot_'+datatype, 'rb'))
    selfbots = pickle.load(open('models_balanced_train/selfdeclaredbots_'+datatype, 'rb'))
    otherbot = pickle.load(open('models_balanced_train/otherbots_'+datatype, 'rb'))
    cyborg = pickle.load(open('models_balanced_train/cyborg_'+datatype, 'rb'))
    botornot = pickle.load(open('models_balanced_train/binary_'+datatype, 'rb'))
    onevsrest = pickle.load(open('models_balanced_train/stacking_model_'+datatype,'rb'))
    multi = return_user_specific_class()
    df['label'] = df['user_id'].map(multi)
    df = df[~df['label'].isin(['otherbots'])]
    if 'max_appearance_of_punc_mark' in df.columns:
        df = df.drop(['max_appearance_of_punc_mark'], axis=1)
    df['label'] = df['label'].map(
        {'human': 0, 'socialbot': 1, 'politicalbot': 2, 'spambot': 3, 'selfdeclaredbots': 4, 'cyborg': 5})
    print('Number of none values:', df.isna().sum().sum())
    print('Number of labelled:' + '\n', df['label'].value_counts())
    X = df.drop(['user_name', 'user_screen_name', 'user_id', 'label', 'time'], axis=1)
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)
    def get_predictions(Xlist,name):
        Xlist = Xlist.to_numpy()
        print (Xlist.shape)
        bothuman = botornot.predict_proba(Xlist)[:,0]
        social = socialbot.predict_proba(Xlist)[:,1]
        pol = politicalbot.predict_proba(Xlist)[:,1]
        spam = spambot.predict_proba(Xlist)[:,1]
        self = selfbots.predict_proba(Xlist)[:,1]
        # other = otherbot.predict_proba(Xlist)[:,1]
        cyb = cyborg.predict_proba(Xlist)[:,1]
        # predictions = np.column_stack([bothuman,social,pol,spam,self,other,cyb])
        predictions = np.column_stack([bothuman,social,pol,spam,self,cyb])
        max_probas = np.max(predictions,axis=1)
        pickle.dump(max_probas,open(name,'wb'))
        min_probas = np.min(predictions, axis=1)
        pickle.dump(min_probas, open('min_probas', 'wb'))
        maxn = np.argmax(predictions,axis=1)
        print ('maxn',maxn)
        return maxn

    def get_prediction_onevsrest(Xlist):
        list = Xlist.to_numpy()
        print(Xlist.shape)
        bothuman = botornot.predict_proba(Xlist)
        one = onevsrest.predict_proba(Xlist)
        print (one[0],bothuman[0])
        one[:,0]=bothuman[:, 0]
        print (one[0])
        print ('we are here',one)
        # predictions = np.column_stack([bothuman, social, pol, spam, self, other, cyb])
        maxn = np.argmax(one, axis=1)
        # maxval = np.max(one, axis=1)
        # print ('maxval',maxval)
        # if bothuman>maxval:
        #     maxn=0
        # else:
        #     maxn=maxn
        # # print (predictions)
        print (maxn)
        return maxn
    y_test = y_test.tolist()
    # y_pred= get_prediction_onevsrest(X_test)
    y_pred = get_predictions(X,'max_probas')
    # y_pred = ys[0]
    # y_max_pred = ys[1]
    # pickle.dump(y_max_pred, open('ymaxpredshort', 'wb'))
    # pickle.dump(y_pred, open('ypredshort', 'wb'))
    print("Accuracy:", metrics.accuracy_score(y, y_pred))
    print("Precision", metrics.precision_score(y, y_pred, average='macro'))
    print("F1-score", metrics.f1_score(y, y_pred, average='macro'))
    print("Recall-score", metrics.recall_score(y, y_pred, average='macro'))
    from sklearn.metrics import classification_report
    print(classification_report(y, y_pred))

def ensemble_as_model(datatype,Xlist):
    Xlist = Xlist.to_numpy()
    print(Xlist.shape)
    socialbot = pickle.load(open('socialbot_' + datatype, 'rb'))
    politicalbot = pickle.load(open('politicalbot_' + datatype, 'rb'))
    spambot = pickle.load(open('spambot_' + datatype, 'rb'))
    selfbots = pickle.load(open('selfdeclaredbots_' + datatype, 'rb'))
    cyborg = pickle.load(open('cyborg_' + datatype, 'rb'))
    botornot = pickle.load(open('binary_' + datatype, 'rb'))
    bothuman = botornot.predict_proba(Xlist)[:, 0]
    social = socialbot.predict_proba(Xlist)[:, 1]
    pol = politicalbot.predict_proba(Xlist)[:, 1]
    spam = spambot.predict_proba(Xlist)[:, 1]
    self = selfbots.predict_proba(Xlist)[:, 1]
    cyb = cyborg.predict_proba(Xlist)[:, 1]
    predictions = np.column_stack([bothuman, social, pol, spam, self, cyb])
    return predictions
