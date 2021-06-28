from datetime import datetime
from feature_tools import get_statistical_results_of_list

def get_user_id(tweets):
    return tweets[0]['user']['id_str']

def get_user_name(tweets):
    return tweets[0]['user']['name']

def get_user_screen_name(tweets):
    return tweets[0]['user']['screen_name']

def get_all_friend_features(rts):
    friends_features=[]
    for r in rts:
        lang=r['retweeted_status']['lang']
        created_at = datetime.strptime(r['retweeted_status']['user']['created_at'],'%a %b %d %H:%M:%S +0000 %Y')
        now = datetime.today()
        delta = now - created_at
        age = delta.days
        num_of_friends = r['retweeted_status']['user']['friends_count']
        num_of_followers = r['retweeted_status']['user']['followers_count']
        num_of_lists = r['retweeted_status']['user']['listed_count']
        num_of_total_tweets = r['retweeted_status']['user']['statuses_count']
        num_of_favs = r['retweeted_status']['user']['favourites_count']
        if (r['retweeted_status']['user']["description"] != None):
            description_length = len(r['retweeted_status']['user']["description"])
        else:
            description_length = 0
        url = r['retweeted_status']['user']['url']
        default_profile = r['retweeted_status']['user']['default_profile']
        default_image = r['retweeted_status']['user']['default_profile_image']
        friend_obj = {"user_id":r['retweeted_status']['user']['id_str'],"lang":lang, "age":age, "friends":num_of_friends, "followers":num_of_followers,
                    "lists":num_of_lists, "statuses":num_of_total_tweets, "favourites":num_of_favs, "description":r['retweeted_status']['user']["description"],
                    "description_length":description_length, "url":url, "default_profile":default_profile,
                    "default_image":default_image}
        friends_features.append(friend_obj)
    return friends_features

def get_unique_retweets_rate(friendsFeatures):
    # print ('get unique retweets rate')
    retweeted = set()
    for i in friendsFeatures:
        retweeted.add(i['user_id'])
    return len(retweeted)/len(friendsFeatures)

def get_num_of_distinct_languages(friendFeatures):
    langs=set()
    checked = []
    for i in friendFeatures:
        if i['user_id'] not in checked:
            checked.append(i['user_id'])
            langs.add(i['lang'])
    return len(langs)

def get_account_age_distribution(friendFeatures):
    ages=[]
    checked = []
    for i in friendFeatures:
        if i['user_id'] not in checked:
            checked.append(i['user_id'])
            ages.append(i['age'])
    return get_statistical_results_of_list(ages)

def get_number_of_friends_distribution(friendFeatures):
    friends = []
    checked = []
    for i in friendFeatures:
        if i['user_id'] not in checked:
            checked.append(i['user_id'])
            friends.append(i['friends'])
    return get_statistical_results_of_list(friends)

def get_number_of_followers_distribution(friendFeatures):
    followers = []
    checked = []
    for i in friendFeatures:
        if i['user_id'] not in checked:
            checked.append(i['user_id'])
            followers.append(i['followers'])
    return get_statistical_results_of_list(followers)

def get_number_of_lists_distribution(friendFeatures):
    lists = []
    checked = []
    for i in friendFeatures:
        if i['user_id'] not in checked:
            checked.append(i['user_id'])
            lists.append(i['lists'])
    return get_statistical_results_of_list(lists)

def get_number_of_statuses_distribution(friendFeatures):
    statuses = []
    checked = []
    for i in friendFeatures:
        if i['user_id'] not in checked:
            checked.append(i['user_id'])
            statuses.append(i['statuses'])
    return get_statistical_results_of_list(statuses)

def get_number_of_favourites_distribution(friendFeatures):
    favourites = []
    checked = []
    for i in friendFeatures:
        if i['user_id'] not in checked:
            checked.append(i['user_id'])
            favourites.append(i['favourites'])
    return get_statistical_results_of_list(favourites)

def get_description_length_distribution(friendFeatures):
    description_lengths = []
    checked = []
    for i in friendFeatures:
        if i['user_id'] not in checked:
            checked.append(i['user_id'])
            description_lengths.append(i['description_length'])
    return get_statistical_results_of_list(description_lengths)

def get_fraction_of_users_with_urls(friendFeatures):
    j=0
    checked = []
    for i in friendFeatures:
        if i['user_id'] not in checked:
            checked.append(i['user_id'])
            if i['url']!=None:
               j+=1
    return j/len(friendFeatures)

def get_fraction_of_users_with_default_profile(friendFeatures):
    j=0
    checked = []
    for i in friendFeatures:
        if i['user_id'] not in checked:
            checked.append(i['user_id'])
            if i['default_profile']==True:
                j+=1
    return j/len(friendFeatures)

def get_fraction_of_users_with_default_image(friendFeatures):
    j=0
    checked = []
    for i in friendFeatures:
        if i['user_id'] not in checked:
            checked.append(i['user_id'])
            if i['default_image']==True:
                j+=1
    return (j/len(friendFeatures))

def get_fraction_of_unique_profile_descriptions(friendFeatures):
    j = 0
    checked = []
    descriptions=set()
    for i in friendFeatures:
        if i['user_id'] not in checked:
            checked.append(i['user_id'])
            if i['description']!=None:
                j+=1
                descriptions.add(i['description'])
    if j==0:
        return 1.0
    else:
        return len(descriptions)/j

