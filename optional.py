import pickle
from pymongo import MongoClient
from tqdm import tqdm

conn = MongoClient('mongodb://--')
db = conn['DBNAME']
col=db['COLLECTION_NAME']

def return_all_datasets():
    datasets = ['Traditional_Spambots_3', 'twittertechnology', 'Verified', 'Pron', 'Gilani',
                'Political', 'News_Agency',
                'Astroturf_political', 'thefakeproject', 'Traditional_Spambots_1', 'Company', 'botwiki',
                'Social_Spambots_3', 'Celebrities', 'Cresci_RTbust',
                'Botometer', 'Vendor', 'Varol', 'Traditional_Spambots_4', 'Social_Spambots_2', 'intertwitter',
                'italian_elections', 'fastfollowerz', 'Social_Spambots_1', 'Genuine_Accounts', 'Cresci_Stock',
                'Midterm', 'Caverlee']
    return datasets

def return_label_per_dataset():
    labels ={'spambot':['Pron','Traditional_Spambots_3','Traditional_Spambots_1','Traditional_Spambots_4','twittertechnology'],
             'socialbot':['Social_Spambots_1','Social_Spambots_2','Social_Spambots_3','fastfollowerz'],
             'politicalbot' :['Astroturf_political','Midterm','Political'],
             'cyborgs' : ['News_Agency','Company','Celebrities'],
             'selfdeclaredbots' : ['botwiki'],
             'otherbots' : ['Botometer','Cresci_RTbust','Gilani','intertwitter','Caverlee']
             }
    return labels

def return_dataset_labels():
    labels ={'Pron':'spambot' , 'Traditional_Spambots_3':'spambot','Traditional_Spambots_1':'spambot','Traditional_Spambots_4':'spambot','twittertechnology':'spambot',
             'Social_Spambots_1':'socialbot', 'Social_Spambots_2':'socialbot', 'Social_Spambots_3':'socialbot', 'fastfollowerz':'socialbot','Cresci_Stock':'socialbot',
             'Astroturf_political':'politicalbot', 'Midterm':'politicalbot', 'Political':'politicalbot','Vendor':'socialbot',
             'News_Agency':'cyborg', 'Company':'cyborg', 'Celebrities':'cyborg','botwiki':'selfdeclaredbots',
             'Botometer':'otherbots', 'Cresci_RTbust':'otherbots', 'Gilani':'otherbots', 'intertwitter':'otherbots', 'Varol':'otherbots','Caverlee':'spambot'}
    return labels

def return_user_specific_class():
    user_dataset = pickle.load(open('pickles/user_dataset_mapping', 'rb'))
    userlabels = pickle.load(open('pickles/user_labels_new', 'rb'))
    bot_datasets = return_dataset_labels()
    final_class = {}
    bot_class = {}
    bots = []
    humans = []
    for u, t in userlabels.items():
        if t == 'human':
            humans.append(u)
            final_class[u] = 'human'
        else:
            bots.append(u)
    j = 0
    for b in bots:
        d = user_dataset[b]
        try:
            clasi = bot_datasets[d]
            final_class[b] = clasi
            # print (b,clasi)
        except KeyError:
            print(d)
    from collections import Counter
    d = dict(Counter(final_class.values()))
    s=0
    for k,v in d.items():
       s=s+int(v)
    return final_class
