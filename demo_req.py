import tweepy
from tqdm import tqdm
import json
from pymongo import MongoClient


conn = MongoClient('mongodb:/--')
db = conn['DBNAME']
col=db['COLLECTION_NAME']

consumer_key = "--"
consumer_secret = "--"
access_key = "--"
access_secret = "--"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)

def on_connect(self):
    # Called initially to connect to the Streaming API
    print("You are now connected to the streaming API.")

class StreamListener(tweepy.StreamListener):
    def on_error(self, status_code):
        # On error - if an error occurs, display the error / status code
        print('An Error has occured: ' + repr(status_code))
        return False

    def on_data(self, data):
        #This is the meat of the script...it connects to your mongoDB and stores the tweet
        try:
            datajson = json.loads(data)
            #grab the 'created_at' data from the Tweet to use for display and change it to Date object
            # created_at = datajson['created_at']
            text = str(datajson['text'].encode('utf8'))
            # dt=datetime.datetime.strptime(created_at, '%a %b %d %H:%M:%S +0000 %Y')
            # verified = datajson['user']['verified']
            # print (verified)
            # datajson['created_at'] = dt
            # print (text)
            #print out a message to the screen that we have collected a tweet
            # print ("Tweet collected at " + str(dt))
            # if verified == True:
            col.insert(datajson)
                # print ('inserted')
            #print a message that the tweet has been inserted into the Db
            #print ('tweet inserted')
        except Exception as e:
           print(e)

def start_stream(WORDS):
    while True:
        try:
            auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
            auth.set_access_token(access_key, access_secret)
            #Set up the listener. The 'wait_on_rate_limit=True' is needed to help with Twitter API rate limiting.
            listener = StreamListener(api=tweepy.API(wait_on_rate_limit=True,wait_on_rate_limit_notify=True,compression=True))
            streamer = tweepy.Stream(auth=auth, listener=listener)
            print("Tracking: " + str(WORDS))
            streamer.filter(track=WORDS)
        except:
            continue

def get_user_info(name):
    try:
        user = api.get_user(name)._json
        return user
    except:
        user = None

def get_timeline_of_user(name):
    tlist = []
    try:
        for page in range(1,6):
            timeline = api.user_timeline(screen_name=name,count=200, tweet_mode="extended", wait_on_rate_limit=True,page=page)
            for t in timeline:
                t = t._json
                tlist.append(t)
    except tweepy.error.TweepError:
        return None
    return tlist

def get_timeline_of_user_and_neighbors(name):
    tweets = get_timeline_of_user(name)
    neighbor_ids = set()
    for t in tweets:
        entities = t['entities']['user_mentions']
        for e in entities:
            neighbor_ids.add(e['id_str'])
    neighbor_tweets = []
    for i in tqdm(neighbor_ids):
        for page in range(1,6):
            try:
                timeline = api.user_timeline(id=i,count=200, tweet_mode="extended", wait_on_rate_limit=True,page=page)
                for status in timeline:
                    status = status._json
                    neighbor_tweets.append(status)
            except:
                break
    return tweets,neighbor_tweets

def get_specific_tweet(id):
    tweet = api.get_status(id)
    return tweet



