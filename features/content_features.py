import math
from nltk import pos_tag, word_tokenize
from twitter_preprocessor import TwitterPreprocessor
from itertools import combinations
from collections import Counter
import string
from features.feature_tools import get_all_texts, get_statistical_results_of_list, get_retweets
import pandas as pd

def get_user_id(tweets):
    return tweets[0]['user']['id_str']

def get_user_name(tweets):
    return tweets[0]['user']['name']

def get_user_screen_name(tweets):
    return tweets[0]['user']['screen_name']

def calculate_entropy(sentence):
    entropy = 0
    # There are 256 possible ASCII characters
    for character_i in range(256):
        Px = sentence.count(chr(character_i)) / len(sentence)
        if Px > 0:
            entropy += - Px * math.log(Px, 2)
    return entropy

def get_text_size_distributions(tweets):
    length=[]
    for t in tweets:
        if 'retweeted_status' not in t:
            try:
                text = t['full_text']
            except KeyError:
                text = t['text']
            p = TwitterPreprocessor(text)
            p.partially_preprocess()
            new = p.text
            numOfWords = len(new.split())
            length.append(numOfWords)
    return get_statistical_results_of_list(length)

def get_text_entropy_distributions(tweets):
    texts = []
    for t in tweets:
        if 'retweeted_status' not in t:
            try:
                text = t['full_text']
            except KeyError:
                text = t['text']
            p = TwitterPreprocessor(text)
            p.partially_preprocess()
            new = p.text
            texts.append(text)
    entropies= []
    for t in texts:
        ent = calculate_entropy(t)
        entropies.append(ent)
    return get_statistical_results_of_list(entropies)

def get_jaccard_sim(str1, str2):
    a = set(str1.split())
    b = set(str2.split())
    c = a.intersection(b)
    try:
        return float(len(c)) / (len(a) + len(b) - len(c))
    except ZeroDivisionError:
        return 0.0

def get_similarities(tweets):
    texts = get_all_texts(tweets)
    similarities = []
    if len(texts)>2:
        combos = combinations(texts, 2)
        for c in combos:
            similarities.append(get_jaccard_sim(c[0], c[1]))
    return get_statistical_results_of_list(similarities)

def get_total_pos_tags(tweets):
    texts = get_all_texts(tweets)
    allTags=[]
    for t in texts:
        p = TwitterPreprocessor(t)
        p.pos_tag_preprocess()
        new = p.text
        text = word_tokenize(new)
        tags = [i[1] for i in pos_tag(text)]
        allTags.extend(tags)
    return allTags
    # print (allTags)

def get_pos_tag_per_tweet(tweets):
    texts = get_all_texts(tweets)
    perTweetTags=[]
    for t in texts:
        p = TwitterPreprocessor(t)
        p.pos_tag_preprocess()
        new = p.text
        text = word_tokenize(new)
        tags = [i[1] for i in pos_tag(text)]
        perTweetTags.append(tags)
    return perTweetTags

def get_proportion_of_POS_tags_per_total_tweets(tweets):
    allTags = get_total_pos_tags(tweets)
    NN, VB, RB, WP, WDT, DT, JJ, PRP, UH = 0, 0, 0, 0, 0, 0, 0, 0, 0
    for tag in allTags:
        if tag.startswith('NN'):
            NN+=1
        if tag.startswith('VB'):
            VB+=1
        if tag.startswith('RB'):
            RB+=1
        if tag.startswith('WDT'):
            WDT+=1
        if tag.startswith('WP'):
            WP+=1
        if tag.startswith('DT'):
            DT+=1
        if tag.startswith('JJ'):
            JJ+=1
        if tag.startswith('PRP'):
            PRP+=1
        if tag.startswith('UH'):
            UH+=1
    return NN/len(tweets),VB/len(tweets),RB/len(tweets),WP/len(tweets),WDT/len(tweets),DT/len(tweets),JJ/len(tweets),PRP/len(tweets),UH/len(tweets)

def get_frequency_of_POS_tag_per_tweet(tweets):
    perTweetTags = get_pos_tag_per_tweet(tweets)
    NN,VB,RB,WP,WDT,DT,JJ,PRP,UH = ([] for i in range(9))
    for tweet in perTweetTags:
        NNc, VBc, RBc, WPc, WDTc, DTc, JJc, PRPc, UHc = 0, 0, 0, 0, 0, 0, 0, 0, 0
        for tag in tweet:
            if tag.startswith('NN'):
                NNc+=1
            if tag.startswith('VB'):
                VBc+=1
            if tag.startswith('RB'):
                RBc+=1
            if tag.startswith('WDT'):
                WDTc+=1
            if tag.startswith('WP'):
                WPc+=1
            if tag.startswith('DT'):
                DTc+=1
            if tag.startswith('JJ'):
                JJc+=1
            if tag.startswith('PRP'):
                PRPc+=1
            if tag.startswith('UH'):
                UHc+=1
        NN.append(NNc)
        VB.append(VBc)
        RB.append(RBc)
        WDT.append(WDTc)
        WP.append(WPc)
        DT.append(DTc)
        JJ.append(JJc)
        PRP.append(PRPc)
        UH.append(UHc)
    return NN,VB,RB,WDT,WP,DT,JJ,PRP,UH

def get_all_punctuation_marks(tweets):
    marks=[]
    all_texts = get_all_texts(tweets)
    for t in all_texts:
        marks.extend([char for char in t if char in string.punctuation])
    return marks

def get_common_marks(tweets):
    allmarks = get_all_punctuation_marks(tweets)
    if len(allmarks) > 0:
        c = Counter(allmarks)
        return c.most_common(1)[0][0],c.most_common(1)[0][1]
    else:
        return '',0

def get_marks_per_tweet(tweets):
    marks_count=[]
    all_texts = get_all_texts(tweets)
    if len(all_texts)>2:
        for t in all_texts:
            marks_count.append(len([char for char in t if char in string.punctuation]))
    return marks_count

def get_marks_distribution(tweets):
    marksPerTweet = get_marks_per_tweet(tweets)
    return get_statistical_results_of_list(marksPerTweet)

def get_tweet_retweet_ratio(tweets):
    ts = 0
    rts = 0
    for t in tweets:
        if 'retweeted_status' in t:
            rts += 1
        else:
            ts += 1
    if rts == 0:
        rts += 1
    ratio = ts / (rts)
    return ts, rts, ratio

def get_total_hashtags_urls_mentions_symbols_media(tweets):
    # print ('get total hashtags urls mentions symbols media')
    tags=[]
    urls=[]
    mentions=[]
    symbols=[]
    media=[]
    for t in tweets:
       entities = t['entities']
       tags.append(len(entities['hashtags']))
       urls.append(len(entities['urls']))
       mentions.append(len(entities['user_mentions']))
       symbols.append(len(entities['symbols']))
       if 'media' in entities:
           media.append(len(entities['media']))
    return tags,urls,mentions,symbols,media

def source_change(tweets):
    sourceSet = set()
    for t in tweets:
        source = t['source']
        sourceSet.add(source)
    if len(sourceSet) > 1:
        return True
    else:
        return False

def source_types(tweets):
    sourceSet = set()
    for t in tweets:
        source = t['source']
        sourceSet.add(source)
    return len(sourceSet)

def get_unique_mentions_rate(tweets):
    mentions = set()
    for t in tweets:
        if 'retweeted_status' not in t:
            entities = t['entities']
            for i in entities['user_mentions']:
                mentions.add(i['id_str'])
    return round(len(mentions)/len(tweets),3)

def get_average_marked_as_favorite(tweets):
    # print ('get average marked as favorite')
    favs = []
    for t in tweets:
        favs.append(t['favorite_count'])
    return get_statistical_results_of_list(favs)

def get_retweeted(tweets):
    rts = []
    for t in tweets:
        if 'retweeted_status' not in t:
            rts.append(t['retweet_count'])
    return get_statistical_results_of_list(rts)

def get_statistics_of_their_retweets(tweets):
    rts=[]
    for t in tweets:
        if 'retweeted_status' in t:
            times_retweeted = t['retweeted_status']['retweet_count']
            rts.append(times_retweeted)
    return get_statistical_results_of_list(rts)
#+++++#
def get_common_text_statistics(tweets):
    texts = get_all_texts(tweets)
    allwords=[]
    for t in texts:
        p = TwitterPreprocessor(t)
        p.partially_preprocess()
        p.lowercase()
        p.remove_single_letter_words()
        p.remove_three_letter_words()
        new = p.text
        text = word_tokenize(new)
        allwords.extend(text)
    # print (allwords)
    lista = pd.Series(allwords).astype('category').cat.codes.values
    return get_statistical_results_of_list(lista)

# import pickle
# ts = pickle.load(open('../pickles/tweets','rb'))
# print (ts)
# print (get_common_text_statistics(ts))