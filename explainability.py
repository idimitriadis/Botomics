import lime
import lime.lime_tabular
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import cross_val_score,RepeatedStratifiedKFold,GridSearchCV,train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from optional import return_user_specific_class
from tqdm import tqdm

def get_all_lime_expl():
    allbots=['spambot', 'socialbot', 'cyborg', 'selfdeclaredbots', 'politicalbot']
    for botclass in tqdm(allbots):
        df = pd.DataFrame(pickle.load(open('final_data_no_rts_v2', 'rb')))
        multi = return_user_specific_class()
        df['label'] = df['user_id'].map(multi)
        df = df[df['label'].isin(['human', botclass])]
        if 'max_appearance_of_punc_mark' in df.columns:
            df = df.drop(['max_appearance_of_punc_mark'], axis=1)
        df['label'] = df['label'].map(
            {'human': 0, botclass: 1})
        X = df.drop(['user_name', 'user_screen_name', 'user_id', 'label', 'time'], axis=1)
        y = df['label']
        print (y.value_counts())
        boolcols=[]
        for c in X.columns:
            if X[c].dtype=='bool':
                boolcols.append(c)
        print (boolcols)
        X[boolcols] = X[boolcols].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=200, random_state=10)
        print (X_test.shape)
        model = pickle.load(open('MODEL_NAME','rb'))
        y_pred = model.predict_proba(X_test)[:,1]
        test_X_imp_df = pd.DataFrame(X_test, columns=X.columns.tolist())

        explainer = lime.lime_tabular.LimeTabularExplainer(
                    X_train.values,
                    training_labels=y_train.values,
                    feature_names=X_train.columns.tolist(),
                    feature_selection="lasso_path",
                    class_names=['human', 'bot'],
                    discretize_continuous=False,
                    discretizer="entropy",
                    kernel_width=5
                )
        print('explainer done...')
        lime_expl = test_X_imp_df.apply(explainer.explain_instance,
                                        predict_fn=model.predict_proba,
                                        num_features=10,
                                        axis=1)
        lime_pred = lime_expl.apply(lambda x: x.local_pred[0])
        pickle.dump(lime_expl, open('lime_expl_'+botclass, 'wb'))

def most_important_features_lime():
    tuples=[]
    allbots=['allbots','spambot', 'socialbot', 'cyborg', 'selfdeclaredbots', 'politicalbot']
    for l in allbots:
        lime_expl = pickle.load(open('lime_expl_'+l, 'rb'))
        features = []
        botftlist = []
        humftlist = []
        for index, row in lime_expl.iteritems():
            # print(row.as_list())
            for i in row.as_list():
                features.append(i)
                if i[1] > 0:
                    botftlist.append(i[0])
                else:
                    humftlist.append(i[0])

        from collections import Counter
        print(l,Counter(botftlist).most_common(5))
        tuples.append(Counter(botftlist).most_common(5))
        # print(Counter(humftlist).most_common(5))

def kernel_vs_mse():
    ks = []
    mss = []
    test_X_imp_df = pd.DataFrame(X_test, columns=X.columns.tolist())
    for k in range(1, 15):
        print(k)
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train.values,
            training_labels=y_train.values,
            feature_names=X_train.columns.tolist(),
            feature_selection="lasso_path",
            class_names=['human', 'bot'],
            discretize_continuous=False,
            discretizer="entropy",
            kernel_width=k
        )
        print('explainer done...')
        lime_expl = test_X_imp_df.apply(explainer.explain_instance,
                                        predict_fn=model.predict_proba,
                                        num_features=10,
                                        axis=1)
        lime_pred = lime_expl.apply(lambda x: x.local_pred[0])
        mse = mean_squared_error(y_pred, lime_pred) ** 0.5
        rsq = r2_score(y_pred, lime_pred)
        print('mse,rsq', mse, rsq)
        ks.append(k)
        mss.append(mse)
    plt.scatter(ks, mss, color='#D81B60')
    plt.xlabel('Kernel Width')
    plt.ylabel('Mean Squared Error')
    plt.show()


