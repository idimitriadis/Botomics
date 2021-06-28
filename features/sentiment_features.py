from vaderSentiment.vaderSentiment import  SentimentIntensityAnalyzer as vds
from features.feature_tools import get_all_texts, get_statistical_results_of_list
from collections import Counter
import emoji

analyzer = vds()

def extract_emojis(s):
    emojis=[c for c in s if c in emoji.UNICODE_EMOJI]
    return emojis

def get_all_emojis(tweets):
    allemojis=[]
    texts = get_all_texts(tweets)
    for t in texts:
        allemojis.extend(extract_emojis(t))
    return allemojis

def tweet_emoji_ratio(tweets):
    texts = get_all_texts(tweets)
    j=0
    for t in texts:
        emojis = extract_emojis(t)
        if len(emojis)>0:
            j+=1
    return j/len(tweets)

def get_most_common_emoji(tweets):
    allemojis = get_all_emojis(tweets)
    if len(allemojis)>0:
        c = Counter(allemojis)
        return c.most_common(1)[0][1]
    else:
        return 0

def get_emojis_per_tweet(tweets):
    emojis_count=[]
    all_texts = get_all_texts(tweets)
    if len(all_texts)>2:
        for t in all_texts:
            emojis_count.append(len(extract_emojis(t)))
            # if len(extract_emojis(t))>5:
            #     print (t,tweets[all_texts.index(t)]['id_str'])
    return get_statistical_results_of_list(emojis_count)

def get_positive_negative_neutral_emojis_per_tweet(tweets):
    neg_emojis_count = []
    pos_emojis_count = []
    neu_emojis_count = []
    all_texts = get_all_texts(tweets)
    if len(all_texts) > 2:
        for t in all_texts:
            neu = 0
            pos = 0
            neg = 0
            emojis = extract_emojis(t)
            if len(emojis)>0:
                for e in emojis:
                    emojiSent = analyzer.polarity_scores(e)
                    if emojiSent['neu']>emojiSent['neg'] and emojiSent['neu']>emojiSent['pos']:
                        neu+=1
                        # print(emojiSent)
                    else:
                        if emojiSent['neg']>emojiSent['pos']:
                            neg+=1
                        else:
                            pos+=1
                neg_emojis_count.append(neg)
                neu_emojis_count.append(neu)
                pos_emojis_count.append(pos)
    return get_statistical_results_of_list(neu_emojis_count),get_statistical_results_of_list(neg_emojis_count),get_statistical_results_of_list(pos_emojis_count)

def get_positive_sentiment_per_tweet(tweets):
    texts = get_all_texts(tweets)
    sentiment = []
    for t in texts:
        sent = analyzer.polarity_scores(t)
        sentiment.append(sent['pos'])
    return get_statistical_results_of_list(sentiment)

def get_negative_sentiment_per_tweet(tweets):
    texts = get_all_texts(tweets)
    sentiment = []
    for t in texts:
        sent = analyzer.polarity_scores(t)
        sentiment.append(sent['neg'])
    return get_statistical_results_of_list(sentiment)

def get_neutral_sentiment_per_tweet(tweets):
    texts = get_all_texts(tweets)
    sentiment = []
    for t in texts:
        sent = analyzer.polarity_scores(t)
        sentiment.append(sent['neg'])
    return get_statistical_results_of_list(sentiment)

