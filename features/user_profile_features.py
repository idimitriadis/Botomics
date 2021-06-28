import re
from datetime import datetime
from textblob import TextBlob
import emoji
import spacy

def get_user_id(user):
	return user['id_str']

def get_user_name(user):
	return user['name']

def get_user_screen_name(user):
	return user['screen_name']

def get_screen_name_length(user):
	return len(user["screen_name"])

def get_name_length(user):
	return len(user["name"])

def get_numbers_count_in_name(user):
	numbers = re.findall(r'\d+', user["name"])
	return len(numbers)

def get_numbers_count_in_screen_name(user):
	numbers = re.findall(r'\d+', user["screen_name"])
	return len(numbers)

def get_followers_count(user):
	return user["followers_count"]

def get_friends_count(user):
	return user["friends_count"]

def get_followers_to_friends(user):
	friends = get_friends_count(user)
	if (friends == 0):
		friends = 1
	return get_followers_count(user) / friends

def get_tweets_count(user):
	return user["statuses_count"]

def get_listed_count(user):
	return user["listed_count"]

def get_favourites_count(user):
	return user["favourites_count"]

def is_default_profile(user):
	return user["default_profile"]

def has_default_profile_image(user):
	return user["default_profile_image"]

def is_verified(user):
	return user["verified"]

def has_location(user):
	if (user["location"] == None):
		return False
	else:
		return True

def has_url(user):
	if (user["url"] == None):
		return False
	else:
		return True

def has_description(user):
	if (user["description"] == None):
		return False
	else:
		return True

def get_description_length(user):
	if (has_description(user)):
		return len(user["description"])
	else:
		return 0

def has_hashtags_in_name(user):
	hashtags = re.findall(r'#\w*', user["name"])
	return len(hashtags) > 0

def has_hashtags_in_description(user):
	if (has_description(user)):
		hashtags = re.findall(r'#\w*', user["description"])
		return len(hashtags) > 0
	else:
		return False

def has_urls_in_description(user):
	if (has_description(user)):
		urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+] |[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', user["description"])
		return len(urls) > 0
	else:
		return False

def has_bot_word_in_name(user):
	# print (user['name'])
	matchObj = re.search('bot', user['name'], flags=re.IGNORECASE)
	if matchObj:
		return True
	else:
		return False

def has_bot_word_in_screen_name(user):
	# print (user['screen_name'])
	matchObj = re.search('bot', user['screen_name'], flags=re.IGNORECASE)
	if matchObj:
		return True
	else:
		return False

def has_bot_word_in_description(user):
	if (has_description(user)):
		matchObj = re.search('bot', user['description'], flags=re.IGNORECASE)
		if matchObj:
			return True
		else:
			return False
	else:
		return False

def get_name_screen_name_similarity(user):
	nlp = spacy.load('en')
	doc1 = nlp(user['name'])
	doc2 = nlp(user['screen_name'])
	# print (doc1)
	return doc1.similarity(doc2)

def get_days_since_creation(user):
	created_at = datetime.strptime(user['created_at'],'%a %b %d %H:%M:%S +0000 %Y')
	now = datetime.today()
	delta = now-created_at
	return (delta.days)

def get_tweet_posting_rate_per_day(user):
	age = get_days_since_creation(user)
	if age==0:
		age+=1
	return get_tweets_count(user)/age

def get_favorite_rate_per_day(user):
	age = get_days_since_creation(user)
	if age == 0:
		age += 1
	return get_favourites_count(user) / age

def get_description_sentiment(user):
	if has_description(user)==True:
		sent = TextBlob(user['description']).polarity
	else:
		sent = 0
	return sent

def extract_emojis(s):
	emojis=[c for c in s if c in emoji.UNICODE_EMOJI]
	return emojis

def get_emojis_in_description(user):
	if has_description(user)==True:
		emojis = extract_emojis(user['description'])
		return len(emojis)
	else:
		return 0






