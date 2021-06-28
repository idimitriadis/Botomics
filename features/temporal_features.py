from datetime import datetime, timedelta
from collections import Counter
import numpy as np
from scipy import stats
from features.feature_tools import get_statistical_results_of_list

def get_user_id(tweets):
    return tweets[0]['user']['id_str']

def get_user_name(tweets):
    return tweets[0]['user']['name']

def get_user_screen_name(tweets):
    return tweets[0]['user']['screen_name']

def get_first_last_day_of_tweets(tweets):
    first = datetime.strptime(tweets[0]['created_at'],'%a %b %d %H:%M:%S +0000 %Y')
    last = datetime.strptime(tweets[-1]['created_at'],'%a %b %d %H:%M:%S +0000 %Y')
    return first,last

def get_total_days(tweets):
    f,l = get_first_last_day_of_tweets(tweets)
    numOfDays = (f-l).days
    return numOfDays

def get_total_hours(tweets):
    f, l = get_first_last_day_of_tweets(tweets)
    numOfHours = (f - l).total_seconds()
    return numOfHours

def get_consecutive_days(tweets):
    f,l = get_first_last_day_of_tweets(tweets)
    dateList = []
    numOfDays = get_total_days(tweets)
    for n in range(numOfDays + 1):
            dateList.append(l.date() + timedelta(n))
    return dateList

def get_max_min_tweets_per_day(tweets):
    # print('get max min tweets per day')
    dates = []
    for t in tweets:
        tweet_date = datetime.strptime(t['created_at'],'%a %b %d %H:%M:%S +0000 %Y').date()
        dates.append(tweet_date)
    date_list = get_consecutive_days(tweets)
    # print (date_list)
    c = Counter(dates)
    if len(date_list)>0:
        for d in date_list:
            if d not in c:
                c[d]=0
        # print (c.values())
        return min(c.values()), max(c.values()), np.mean(list(c.values())), np.median(list(c.values())), np.std(list(c.values())), \
               stats.skew(list(c.values())), stats.kurtosis(list(c.values())), stats.entropy(list(c.values())),c
    else:
        # print('get_max_min_tweets_per_day')
        return 0,0,0,0,0,0,0,0,c

def get_max_min_tweets_per_hour(tweets):
    # print ('get_max_min_tweets_per_hour')
    dates = []
    for t in tweets:
        tweet_date = datetime.strptime(t['created_at'],'%a %b %d %H:%M:%S +0000 %Y')
        tweet_date = tweet_date.replace(minute=0,second=0)
        dates.append(tweet_date)
    date_list = get_consecutive_hours(tweets)
    # print (date_list)
    c = Counter(dates)
    # print (c)
    if len(date_list)>0:
        for d in date_list:
            if d not in c:
                c[d]=0
        return min(c.values()), max(c.values()), np.mean(list(c.values())), np.median(list(c.values())), np.std(list(c.values())), \
               stats.skew(list(c.values())), stats.kurtosis(list(c.values())), stats.entropy(list(c.values())),c
    else:
        # print ('error get_max_min_tweets_per_hour')
        return 0,0,0,0,0,0,0,0,c

def get_consecutive_days_of_no_activity(tweets):
    minV, maxV, meanV, medianV, stdV, \
    skewV, kurtV, entV, c= get_max_min_tweets_per_day(tweets)
    alldates = get_consecutive_days(tweets)
    i = 0
    maximum = 0
    dat = []
    if len(alldates) > 0:
        for d in alldates:
           if c[d]==0:
               dat.append(d)
               i+=1
           else:
               if i > maximum:
                   maximum = i
               i=0
        if maximum == 0:
            maximum = i
        # print ('maximun days of no activity',maximum)
        # print (dat)
        return maximum
    else:
        return 0

def get_consecutive_days_of_activity(tweets):
    # print ('get_consecutive_days_of_activity')
    minV, maxV, meanV, medianV, stdV, \
    skewV, kurtV, entV, c = get_max_min_tweets_per_day(tweets)
    alldates = get_consecutive_days(tweets)
    i = 0
    maximum = 0
    dat = []
    if len(alldates) > 0:
        for d in alldates:
           if c[d]>0:
               dat.append(d)
               i+=1
           else:
               if i > maximum:
                   maximum = i
               i=0
        if maximum == 0:
            maximum = i
        # print ('maximum days of activity',maximum)
        # print (dat)
        return maximum
    else:
        return maximum

def get_consecutive_hours(tweets):
    f, l = get_first_last_day_of_tweets(tweets)
    dateList = []
    l = l.replace(minute=0,second=0)
    numOfHours = int(get_total_hours(tweets)/3600)
    for n in range(numOfHours + 1):
        dateList.append(l + (timedelta(hours=n)))
    return dateList

def get_consecutive_hours_of_no_activity(tweets):
    # print ('get_consecutive_hours_of_no_activity')
    minV, maxV, meanV, medianV, stdV, \
    skewV, kurtV, entV, c = get_max_min_tweets_per_hour(tweets)
    alldates = get_consecutive_hours(tweets)
    i = 0
    maximum = 0
    dat = []
    if len(alldates)>0:
        for d in alldates:
            if c[d] == 0:
                dat.append(d)
                i += 1
            else:
                if i > maximum:
                    maximum = i
                i = 0
        if maximum == 0:
            maximum = i
        # print('max consecutive hours of no activiyt',maximum)
        # print(dat)
        return maximum
    else:
        return maximum
        # print ('error in get consecutive hours of no activity')

def get_consecutive_hours_of_activity(tweets):
    # print ('get_consecutive_hours_of_activity')
    minV, maxV, meanV, medianV, stdV, \
    skewV, kurtV, entV, c = get_max_min_tweets_per_hour(tweets)
    alldates = get_consecutive_hours(tweets)
    i = 0
    maximum = 0
    dat = []
    if len(alldates)>0:
        for d in alldates:
            if c[d] > 0:
                dat.append(d)
                i += 1
            else:
                if i > maximum:
                    maximum = i
                i = 0
        if maximum == 0:
            maximum = i
        # print('max consecutive hours of activiyt',maximum)
        # print(dat)
        return maximum
    else:
        return maximum
        # print ('error in get_consecutive_hours_of_activity')

def get_average_time_between_tweets(tweets):
    # print ('get average time between tweets')
    gapList = []
    for i in range(0, len(tweets) - 1):
        first = datetime.strptime(tweets[i]['created_at'],
                                  '%a %b %d %H:%M:%S +0000 %Y')
        second = datetime.strptime(tweets[i + 1]['created_at'],
                                   '%a %b %d %H:%M:%S +0000 %Y')
        gap = ((first - second).seconds)
        gapList.append(gap)
    return get_statistical_results_of_list(gapList)

def get_max_occurence_of_same_gap(tweets):
    # print ('get average time between tweets')
    gapList = []
    if len(tweets) > 2:
        for i in range(0, len(tweets) - 1):
            first = datetime.strptime(tweets[i]['created_at'],
                                      '%a %b %d %H:%M:%S +0000 %Y')
            second = datetime.strptime(tweets[i + 1]['created_at'],
                                       '%a %b %d %H:%M:%S +0000 %Y')
            gap = ((first - second).seconds)
            gapList.append(gap)
        c = Counter(gapList)
        # print (gapList)
        max_occ = c.most_common(1)[0][1]
    else:
        max_occ = 0
    return max_occ

