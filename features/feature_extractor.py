import content_features as cf
import friends_features as ff
import sentiment_features as sf
import temporal_features as tf
import user_profile_features as uf
import network_features as nf
import feature_tools as tools
import pickle


#CAN BE USED FOR ALL ACCOUNTS - DESPITE THE EXISTENCE OF TWEETS/RETWEETS#
def get_user_vector(user):
	user_vector = {
        "user_id" : uf.get_user_id(user),
        "user_name" : uf.get_user_name(user),
        "user_screen_name" : uf.get_user_screen_name(user),
        "followers_count": uf.get_followers_count(user),
		"followees_count": uf.get_friends_count(user),
		"followers_to_friends": uf.get_followers_to_friends(user),
		"tweets_count": uf.get_tweets_count(user),
		"listed_count": uf.get_listed_count(user),
		"favorites_count": uf.get_favourites_count(user),
		"default_profile": uf.is_default_profile(user),
		"default_profile_image": uf.has_default_profile_image(user),
		"verified": uf.is_verified(user),
		"location": uf.has_location(user),
		"url": uf.has_url(user),
		"description": uf.has_description(user),
		"name_length": uf.get_name_length(user),
		"screen_name_length": uf.get_screen_name_length(user),
		"description_length": uf.get_description_length(user),
		"numerics_in_name_count": uf.get_numbers_count_in_name(user),
		"numerics_in_screen_name_count": uf.get_numbers_count_in_screen_name(user),
		"hashtags_in_name": uf.has_hashtags_in_name(user),
		"hashtags_in_description": uf.has_hashtags_in_description(user),
		"urls_in_description": uf.has_urls_in_description(user),
		"bot_word_in_name": uf.has_bot_word_in_name(user),
		"bot_word_in_screen_name": uf.has_bot_word_in_screen_name(user),
		"bot_word_in_description": uf.has_bot_word_in_description(user),
        "tweet_posting_rate_per_day":uf.get_tweet_posting_rate_per_day(user),
        "favorite_rate_per_day":uf.get_favorite_rate_per_day(user),
        "name_screen_name_similarity":uf.get_name_screen_name_similarity(user),
        "description_sentiment":uf.get_description_sentiment(user),
        "description_emojis" : uf.get_emojis_in_description(user)
    }

	return user_vector

#CAN ONLY BE USED WITH ACCOUNTS THAT HAVE AT LEAST 1 RETWEET
def get_friend_feature_vector(tweets):
	retweets= tools.get_retweets(tweets)
	friend_feats = ff.get_all_friend_features(retweets)
	age_distros = ff.get_account_age_distribution(friend_feats)
	friends_distros = ff.get_number_of_friends_distribution(friend_feats)
	followers_distros = ff.get_number_of_followers_distribution(friend_feats)
	lists_distros = ff.get_number_of_lists_distribution(friend_feats)
	statuses_distros = ff.get_number_of_statuses_distribution(friend_feats)
	favourites_distros = ff.get_number_of_favourites_distribution(friend_feats)
	description_lengths_distros = ff.get_description_length_distribution(friend_feats)
	friends_vector = {
		"user_id": tools.get_user_id(tweets),
		"user_name": tools.get_user_name(tweets),
		"user_screen_name": tools.get_user_screen_name(tweets),
		"unique_retweet_rate": ff.get_unique_retweets_rate(friend_feats),
		"distinct_langs" : ff.get_num_of_distinct_languages(friend_feats),
		"min_age" : age_distros[0],
		"max_age" : age_distros[1],
		"mean_age" : age_distros[2],
		"median_age" : age_distros[3],
		"std_age" : age_distros[4],
		"skew_age" : age_distros[5],
		"kurtos_age" : age_distros[6],
		"entropy_age" : age_distros[7],
		"min_friends": friends_distros[0],
		"max_friends": friends_distros[1],
		"mean_friends": friends_distros[2],
		"median_friends": friends_distros[3],
		"std_friends": friends_distros[4],
		"skew_friends": friends_distros[5],
		"kurtos_friends": friends_distros[6],
		"entropy_friends": friends_distros[7],
		"min_followers": followers_distros[0],
		"max_followers": followers_distros[1],
		"mean_followers": followers_distros[2],
		"median_followers": followers_distros[3],
		"std_followers": followers_distros[4],
		"skew_followers": followers_distros[5],
		"kurtos_followers": followers_distros[6],
		"entropy_followers": followers_distros[7],
		"min_lists": lists_distros[0],
		"max_lists": lists_distros[1],
		"mean_lists": lists_distros[2],
		"median_lists": lists_distros[3],
		"std_lists": lists_distros[4],
		"skew_lists": lists_distros[5],
		"kurtos_lists": lists_distros[6],
		"entropy_lists": lists_distros[7],
		"min_statuses": statuses_distros[0],
		"max_statuses": statuses_distros[1],
		"mean_statuses": statuses_distros[2],
		"median_statuses": statuses_distros[3],
		"std_statuses": statuses_distros[4],
		"skew_statuses": statuses_distros[5],
		"kurtos_statuses": statuses_distros[6],
		"entropy_statuses": statuses_distros[7],
		"min_favourites": favourites_distros[0],
		"max_favourites": favourites_distros[1],
		"mean_favourites": favourites_distros[2],
		"median_favourites": favourites_distros[3],
		"std_favourites": favourites_distros[4],
		"skew_favourites": favourites_distros[5],
		"kurtos_favourites": favourites_distros[6],
		"entropy_favourites": favourites_distros[7],
		"min_description_lengths": description_lengths_distros[0],
		"max_description_lengths": description_lengths_distros[1],
		"mean_description_lengths": description_lengths_distros[2],
		"median_description_lengths": description_lengths_distros[3],
		"std_description_lengths": description_lengths_distros[4],
		"skew_description_lengths": description_lengths_distros[5],
		"kurtos_description_lengths": description_lengths_distros[6],
		"entropy_description_lengths": description_lengths_distros[7],
		"fraction_of_users_with_urls" : ff.get_fraction_of_users_with_urls(friend_feats),
		"fraction_of_users_with_default_profile" : ff.get_fraction_of_users_with_default_profile(friend_feats),
		"fraction_of_users_with_default_image" : ff.get_fraction_of_users_with_default_image(friend_feats),
		"fraction_of_unique_profile_descriptions" : ff.get_fraction_of_unique_profile_descriptions(friend_feats)
	}
	return friends_vector

#CAN BE USED IF USER HAS RETWEETS AND TWEETS
def get_temporal_features_vector(tweets):
	retweets = tools.get_retweets(tweets)
	only_tweets = tools.get_only_tweets(tweets)
	total_tweets_per_day_distro = tf.get_max_min_tweets_per_day(tweets)
	total_tweets_per_hour_distro = tf.get_max_min_tweets_per_hour(tweets)
	tweets_per_day_distro = tf.get_max_min_tweets_per_day(only_tweets)
	tweets_per_hour_distro = tf.get_max_min_tweets_per_hour(only_tweets)
	retweets_per_day_distro = tf.get_max_min_tweets_per_day(retweets)
	retweets_per_hour_distro = tf.get_max_min_tweets_per_hour(retweets)
	avg_time_between_tweets_distro = tf.get_average_time_between_tweets(only_tweets)
	avg_time_between_retweets_distro = tf.get_average_time_between_tweets(retweets)
	temporal_features_vector={
		"user_id": tools.get_user_id(tweets),
		"user_name": tools.get_user_name(tweets),
		"user_screen_name": tools.get_user_screen_name(tweets),
		"total_min_tweets_per_day": total_tweets_per_day_distro[0],
		"total_max_tweets_per_day": total_tweets_per_day_distro[1],
		"total_mean_tweets_per_day": total_tweets_per_day_distro[2],
		"total_median_tweets_per_day": total_tweets_per_day_distro[3],
		"total_std_tweets_per_day": total_tweets_per_day_distro[4],
		"total_skew_tweets_per_day": total_tweets_per_day_distro[5],
		"total_kurt_tweets_per_day": total_tweets_per_day_distro[6],
		"total_entropy_tweets_per_day": total_tweets_per_day_distro[7],
		"total_min_tweets_per_hour": total_tweets_per_hour_distro[0],
		"total_max_tweets_per_hour": total_tweets_per_hour_distro[1],
		"total_mean_tweets_per_hour": total_tweets_per_hour_distro[2],
		"total_median_tweets_per_hour": total_tweets_per_hour_distro[3],
		"total_std_tweets_per_hour": total_tweets_per_hour_distro[4],
		"total_skew_tweets_per_hour": total_tweets_per_hour_distro[5],
		"total_kurt_tweets_per_hour": total_tweets_per_hour_distro[6],
		"total_entropy_tweets_per_hour": total_tweets_per_hour_distro[7],
		"min_tweets_per_day" : tweets_per_day_distro[0],
		"max_tweets_per_day" : tweets_per_day_distro[1],
		"mean_tweets_per_day" : tweets_per_day_distro[2],
		"median_tweets_per_day" : tweets_per_day_distro[3],
		"std_tweets_per_day" : tweets_per_day_distro[4],
		"skew_tweets_per_day" : tweets_per_day_distro[5],
		"kurt_tweets_per_day" : tweets_per_day_distro[6],
		"entropy_tweets_per_day" : tweets_per_day_distro[7],
		"min_tweets_per_hour": tweets_per_hour_distro[0],
		"max_tweets_per_hour": tweets_per_hour_distro[1],
		"mean_tweets_per_hour": tweets_per_hour_distro[2],
		"median_tweets_per_hour": tweets_per_hour_distro[3],
		"std_tweets_per_hour": tweets_per_hour_distro[4],
		"skew_tweets_per_hour": tweets_per_hour_distro[5],
		"kurt_tweets_per_hour": tweets_per_hour_distro[6],
		"entropy_tweets_per_hour": tweets_per_hour_distro[7],
		"min_retweets_per_day": retweets_per_day_distro[0],
		"max_retweets_per_day": retweets_per_day_distro[1],
		"mean_retweets_per_day": retweets_per_day_distro[2],
		"median_retweets_per_day": retweets_per_day_distro[3],
		"std_retweets_per_day": retweets_per_day_distro[4],
		"skew_retweets_per_day": retweets_per_day_distro[5],
		"kurt_retweets_per_day": retweets_per_day_distro[6],
		"entropy_retweets_per_day": retweets_per_day_distro[7],
		"min_retweets_per_hour": retweets_per_hour_distro[0],
		"max_retweets_per_hour": retweets_per_hour_distro[1],
		"mean_retweets_per_hour": retweets_per_hour_distro[2],
		"median_retweets_per_hour": retweets_per_hour_distro[3],
		"std_retweets_per_hour": retweets_per_hour_distro[4],
		"skew_retweets_per_hour": retweets_per_hour_distro[5],
		"kurt_retweets_per_hour": retweets_per_hour_distro[6],
		"entropy_retweets_per_hour": retweets_per_hour_distro[7],
		"consecutive_days_of_no_activity" : tf.get_consecutive_days_of_no_activity(tweets),
		"consecutive_days_of_activity" : tf.get_consecutive_days_of_activity(tweets),
		"consecutive_hours_of_no_activity" : tf.get_consecutive_hours_of_no_activity(tweets),
		"consecutive_hours_of_activity" : tf.get_consecutive_hours_of_activity(tweets),
		"min_avg_time_between_tweets" : avg_time_between_tweets_distro[0],
		"max_avg_time_between_tweets" : avg_time_between_tweets_distro[1],
		"mean_avg_time_between_tweets" : avg_time_between_tweets_distro[2],
		"median_avg_time_between_tweets" : avg_time_between_tweets_distro[3],
		"std_avg_time_between_tweets" : avg_time_between_tweets_distro[4],
		"skew_avg_time_between_tweets" : avg_time_between_tweets_distro[5],
		"kurt_avg_time_between_tweets" : avg_time_between_tweets_distro[6],
		"entropy_avg_time_between_tweets" : avg_time_between_tweets_distro[7],
		"min_avg_time_between_retweets": avg_time_between_retweets_distro[0],
		"max_avg_time_between_retweets": avg_time_between_retweets_distro[1],
		"mean_avg_time_between_retweets": avg_time_between_retweets_distro[2],
		"median_avg_time_between_retweets": avg_time_between_retweets_distro[3],
		"std_avg_time_between_retweets": avg_time_between_retweets_distro[4],
		"skew_avg_time_between_retweets": avg_time_between_retweets_distro[5],
		"kurt_avg_time_between_retweets": avg_time_between_retweets_distro[6],
		"entropy_avg_time_between_retweets": avg_time_between_retweets_distro[7],
		"max_occurence_of_same_gap_in_seconds" : tf.get_max_occurence_of_same_gap(tweets)
	}

	return temporal_features_vector

#CAN BE USED REGARDLESS THE TYPE OF CONTENT THE USER HAS (EITHER TWEETS OR RETWEETS)
def get_temporal_features_vector_only_twts(tweets):
	tweets_per_day_distro = tf.get_max_min_tweets_per_day(tweets)
	tweets_per_hour_distro = tf.get_max_min_tweets_per_hour(tweets)
	avg_time_between_tweets_distro = tf.get_average_time_between_tweets(tweets)
	temporal_features_vector={
		"user_id": tools.get_user_id(tweets),
		"user_name": tools.get_user_name(tweets),
		"user_screen_name": tools.get_user_screen_name(tweets),
		"min_tweets_per_day" : tweets_per_day_distro[0],
		"max_tweets_per_day" : tweets_per_day_distro[1],
		"mean_tweets_per_day" : tweets_per_day_distro[2],
		"median_tweets_per_day" : tweets_per_day_distro[3],
		"std_tweets_per_day" : tweets_per_day_distro[4],
		"skew_tweets_per_day" : tweets_per_day_distro[5],
		"kurt_tweets_per_day" : tweets_per_day_distro[6],
		"entropy_tweets_per_day" : tweets_per_day_distro[7],
		"min_tweets_per_hour": tweets_per_hour_distro[0],
		"max_tweets_per_hour": tweets_per_hour_distro[1],
		"mean_tweets_per_hour": tweets_per_hour_distro[2],
		"median_tweets_per_hour": tweets_per_hour_distro[3],
		"std_tweets_per_hour": tweets_per_hour_distro[4],
		"skew_tweets_per_hour": tweets_per_hour_distro[5],
		"kurt_tweets_per_hour": tweets_per_hour_distro[6],
		"entropy_tweets_per_hour": tweets_per_hour_distro[7],
		"consecutive_days_of_no_activity" : tf.get_consecutive_days_of_no_activity(tweets),
		"consecutive_days_of_activity" : tf.get_consecutive_days_of_activity(tweets),
		"consecutive_hours_of_no_activity" : tf.get_consecutive_hours_of_no_activity(tweets),
		"consecutive_hours_of_activity" : tf.get_consecutive_hours_of_activity(tweets),
		"min_avg_time_between_tweets" : avg_time_between_tweets_distro[0],
		"max_avg_time_between_tweets" : avg_time_between_tweets_distro[1],
		"mean_avg_time_between_tweets" : avg_time_between_tweets_distro[2],
		"median_avg_time_between_tweets" : avg_time_between_tweets_distro[3],
		"std_avg_time_between_tweets" : avg_time_between_tweets_distro[4],
		"skew_avg_time_between_tweets" : avg_time_between_tweets_distro[5],
		"kurt_avg_time_between_tweets" : avg_time_between_tweets_distro[6],
		"entropy_avg_time_between_tweets" : avg_time_between_tweets_distro[7],
		"max_occurence_of_same_gap_in_seconds" : tf.get_max_occurence_of_same_gap(tweets)
	}

	return temporal_features_vector

#APPLIES IF USER HAS TWEETS AND RETWEETS#
def get_content_feature_vector(tweets):
	retweets = tools.get_retweets(tweets)
	only_tweets = tools.get_only_tweets(tweets)
	if len(only_tweets)==0:
		only_tweets = retweets
	text_size_distro = cf.get_text_size_distributions(only_tweets)
	text_entropy_distributions = cf.get_text_entropy_distributions(only_tweets)
	similarities_distro = cf.get_similarities(only_tweets)
	text_distro = cf.get_common_text_statistics(only_tweets)
	pros_tag_proportions = cf.get_proportion_of_POS_tags_per_total_tweets(only_tweets)
	NN, VB, RB, WDT, WP, DT, JJ, PRP, UH = cf.get_frequency_of_POS_tag_per_tweet(only_tweets)
	NN_distro  = tools.get_statistical_results_of_list(NN)
	VB_distro = tools.get_statistical_results_of_list(VB)
	RB_distro = tools.get_statistical_results_of_list(RB)
	WDT_distro = tools.get_statistical_results_of_list(WDT)
	WP_distro = tools.get_statistical_results_of_list(WP)
	DT_distro = tools.get_statistical_results_of_list(DT)
	JJ_distro = tools.get_statistical_results_of_list(JJ)
	PRP_distro = tools.get_statistical_results_of_list(PRP)
	UH_distro = tools.get_statistical_results_of_list(UH)
	marks_distro = cf.get_marks_distribution(only_tweets)
	tags, urls, mentions, symbols, media = cf.get_total_hashtags_urls_mentions_symbols_media(tweets)
	tags_distro  = tools.get_statistical_results_of_list(tags)
	urls_distro = tools.get_statistical_results_of_list(urls)
	mentions_distro = tools.get_statistical_results_of_list(mentions)
	symbols_distro = tools.get_statistical_results_of_list(symbols)
	media_distro = tools.get_statistical_results_of_list(media)
	favs_distro = cf.get_average_marked_as_favorite(only_tweets)
	rts_distro = cf.get_retweeted(only_tweets)
	others_rts_distro = cf.get_statistics_of_their_retweets(retweets)
	content_vector = {
		"user_id": tools.get_user_id(tweets),
		"user_name": tools.get_user_name(tweets),
		"user_screen_name": tools.get_user_screen_name(tweets),
		"min_text_size": text_size_distro[0],
		"max_text_size": text_size_distro[1],
		"mean_text_size": text_size_distro[2],
		"median_text_size": text_size_distro[3],
		"std_text_size": text_size_distro[4],
		"skew_text_size": text_size_distro[5],
		"kurt_text_size": text_size_distro[6],
		"entropy_text_size": text_size_distro[7],
		"min_text_entropy": text_entropy_distributions[0],
		"max_text_entropy": text_entropy_distributions[1],
		"mean_text_entropy": text_entropy_distributions[2],
		"median_text_entropy": text_entropy_distributions[3],
		"std_text_entropy": text_entropy_distributions[4],
		"skew_text_entropy": text_entropy_distributions[5],
		"kurt_text_entropy": text_entropy_distributions[6],
		"entropy_text_entropy": text_entropy_distributions[7],
		"min_similarity": similarities_distro[0],
		"max_similarity": similarities_distro[1],
		"mean_similarity": similarities_distro[2],
		"median_similarity": similarities_distro[3],
		"std_similarity": similarities_distro[4],
		"skew_similarity": similarities_distro[5],
		"kurt_similarity": similarities_distro[6],
		"entropy_similarity": similarities_distro[7],
		"NN_proportion" : pros_tag_proportions[0],
		"VB_proportion" : pros_tag_proportions[1],
		"RB_proportion" : pros_tag_proportions[2],
		"WP_proportion" : pros_tag_proportions[3],
		"WDT_proportion" : pros_tag_proportions[4],
		"DT_proportion" : pros_tag_proportions[5],
		"JJ_proportion" : pros_tag_proportions[6],
		"PRP_proportion" : pros_tag_proportions[7],
		"UH_proportion" : pros_tag_proportions[8],
		"min_NN": NN_distro[0],
		"max_NN": NN_distro[1],
		"mean_NN": NN_distro[2],
		"median_NN": NN_distro[3],
		"std_NN": NN_distro[4],
		"skew_NN": NN_distro[5],
		"kurt_NN": NN_distro[6],
		"entropy_NN": NN_distro[7],
		"min_VB": VB_distro[0],
		"max_VB": VB_distro[1],
		"mean_VB": VB_distro[2],
		"median_VB": VB_distro[3],
		"std_VB": VB_distro[4],
		"skew_VB": VB_distro[5],
		"kurt_VB": VB_distro[6],
		"entropy_VB": VB_distro[7],
		"min_RB": RB_distro[0],
		"max_RB": RB_distro[1],
		"mean_RB": RB_distro[2],
		"median_RB": RB_distro[3],
		"std_RB": RB_distro[4],
		"skew_RB": RB_distro[5],
		"kurt_RB": RB_distro[6],
		"entropy_RB": RB_distro[7],
		"min_WP": WP_distro[0],
		"max_WP": WP_distro[1],
		"mean_WP": WP_distro[2],
		"median_WP": WP_distro[3],
		"std_WP": WP_distro[4],
		"skew_WP": WP_distro[5],
		"kurt_WP": WP_distro[6],
		"entropy_WP": WP_distro[7],
		"min_DT": DT_distro[0],
		"max_DT": DT_distro[1],
		"mean_DT": DT_distro[2],
		"median_DT": DT_distro[3],
		"std_DT": DT_distro[4],
		"skew_DT": DT_distro[5],
		"kurt_DT": DT_distro[6],
		"entropy_DT": DT_distro[7],
		"min_WDT": WDT_distro[0],
		"max_WDT": WDT_distro[1],
		"mean_WDT": WDT_distro[2],
		"median_WDT": WDT_distro[3],
		"std_WDT": WDT_distro[4],
		"skew_WDT": WDT_distro[5],
		"kurt_WDT": WDT_distro[6],
		"entropy_WDT": WDT_distro[7],
		"min_JJ": JJ_distro[0],
		"max_JJ": JJ_distro[1],
		"mean_JJ": JJ_distro[2],
		"median_JJ": JJ_distro[3],
		"std_JJ": JJ_distro[4],
		"skew_JJ": JJ_distro[5],
		"kurt_JJ": JJ_distro[6],
		"entropy_JJ": JJ_distro[7],
		"min_PRP": PRP_distro[0],
		"max_PRP": PRP_distro[1],
		"mean_PRP": PRP_distro[2],
		"median_PRP": PRP_distro[3],
		"std_PRP": PRP_distro[4],
		"skew_PRP": PRP_distro[5],
		"kurt_PRP": PRP_distro[6],
		"entropy_PRP": PRP_distro[7],
		"min_UH": UH_distro[0],
		"max_UH": UH_distro[1],
		"mean_UH": UH_distro[2],
		"median_UH": UH_distro[3],
		"std_UH": UH_distro[4],
		"skew_UH": UH_distro[5],
		"kurt_UH": UH_distro[6],
		"entropy_UH": UH_distro[7],
		"max_appearance_of_punc_mark" : cf.get_common_marks(only_tweets),
		"min_marks": marks_distro[0],
		"max_marks": marks_distro[1],
		"mean_marks": marks_distro[2],
		"median_marks": marks_distro[3],
		"std_marks": marks_distro[4],
		"skew_marks": marks_distro[5],
		"kurt_marks": marks_distro[6],
		"entropy_marks": marks_distro[7],
		"tweet_retweet_ratio" : cf.get_tweet_retweet_ratio(tweets)[2],
		"min_tags": tags_distro[0],
		"max_tags": tags_distro[1],
		"mean_tags": tags_distro[2],
		"median_tags": tags_distro[3],
		"std_tags": tags_distro[4],
		"skew_tags": tags_distro[5],
		"kurt_tags": tags_distro[6],
		"entropy_tags": tags_distro[7],
		"min_urls": urls_distro[0],
		"max_urls": urls_distro[1],
		"mean_urls": urls_distro[2],
		"median_urls": urls_distro[3],
		"std_urls": urls_distro[4],
		"skew_urls": urls_distro[5],
		"kurt_urls": urls_distro[6],
		"entropy_urls": urls_distro[7],
		"min_mentions": mentions_distro[0],
		"max_mentions": mentions_distro[1],
		"mean_mentions": mentions_distro[2],
		"median_mentions": mentions_distro[3],
		"std_mentions": mentions_distro[4],
		"skew_mentions": mentions_distro[5],
		"kurt_mentions": mentions_distro[6],
		"entropy_mentions" : mentions_distro[7],
		"min_symbols": symbols_distro[0],
		"max_symbols": symbols_distro[1],
		"mean_symbols": symbols_distro[2],
		"median_symbols": symbols_distro[3],
		"std_symbols": symbols_distro[4],
		"skew_symbols": symbols_distro[5],
		"kurt_symbols": symbols_distro[6],
		"entropy_symbols": symbols_distro[7],
		"min_media": media_distro[0],
		"max_media": media_distro[1],
		"mean_media": media_distro[2],
		"median_media": media_distro[3],
		"std_media": media_distro[4],
		"skew_media": media_distro[5],
		"kurt_media": media_distro[6],
		"entropy_media": media_distro[7],
		"source_change" : cf.source_change(tweets),
		"source_types" : cf.source_types(tweets),
		"unique_mentions_rate" : cf.get_unique_mentions_rate(tweets),
		"min_favs" : favs_distro[0],
		"max_favs": favs_distro[1],
		"mean_favs": favs_distro[2],
		"median_favs": favs_distro[3],
		"std_favs": favs_distro[4],
		"skew_favs": favs_distro[5],
		"kurt_favs": favs_distro[6],
		"entropy_favs": favs_distro[7],
		"min_rts": rts_distro[0],
		"max_rts": rts_distro[1],
		"mean_rts": rts_distro[2],
		"median_rts": rts_distro[3],
		"std_rts": rts_distro[4],
		"skew_rts": rts_distro[5],
		"kurt_rts": rts_distro[6],
		"entropy_rts": rts_distro[7],
		"min_tokens": text_distro[0],
		"max_tokens": text_distro[1],
		"mean_tokens": text_distro[2],
		"median_tokens": text_distro[3],
		"std_tokens": text_distro[4],
		"skew_tokens": text_distro[5],
		"kurt_tokens": text_distro[6],
		"entropy_tokens": text_distro[7],
		"min_others_rts": others_rts_distro[0],
		"max_others_rts": others_rts_distro[1],
		"mean_others_rts": others_rts_distro[2],
		"median_others_rts": others_rts_distro[3],
		"std_others_rts": others_rts_distro[4],
		"skew_others_rts": others_rts_distro[5],
		"kurt_others_rts": others_rts_distro[6],
		"entropy_others_rts": others_rts_distro[7]
	}

	return content_vector

#APPLIES TO TO ALL - REGARDLESS THE TYPE OF CONTENT
def get_content_feature_vector_twts(tweets):
	text_size_distro = cf.get_text_size_distributions(tweets)
	text_entropy_distributions = cf.get_text_entropy_distributions(tweets)
	similarities_distro = cf.get_similarities(tweets)
	text_distro = cf.get_common_text_statistics(tweets)
	pros_tag_proportions = cf.get_proportion_of_POS_tags_per_total_tweets(tweets)
	NN, VB, RB, WDT, WP, DT, JJ, PRP, UH = cf.get_frequency_of_POS_tag_per_tweet(tweets)
	NN_distro  = tools.get_statistical_results_of_list(NN)
	VB_distro = tools.get_statistical_results_of_list(VB)
	RB_distro = tools.get_statistical_results_of_list(RB)
	WDT_distro = tools.get_statistical_results_of_list(WDT)
	WP_distro = tools.get_statistical_results_of_list(WP)
	DT_distro = tools.get_statistical_results_of_list(DT)
	JJ_distro = tools.get_statistical_results_of_list(JJ)
	PRP_distro = tools.get_statistical_results_of_list(PRP)
	UH_distro = tools.get_statistical_results_of_list(UH)
	marks_distro = cf.get_marks_distribution(tweets)
	tags, urls, mentions, symbols, media = cf.get_total_hashtags_urls_mentions_symbols_media(tweets)
	tags_distro  = tools.get_statistical_results_of_list(tags)
	urls_distro = tools.get_statistical_results_of_list(urls)
	mentions_distro = tools.get_statistical_results_of_list(mentions)
	symbols_distro = tools.get_statistical_results_of_list(symbols)
	media_distro = tools.get_statistical_results_of_list(media)
	favs_distro = cf.get_average_marked_as_favorite(tweets)
	rts_distro = cf.get_retweeted(tweets)
	content_vector = {
		"user_id": tools.get_user_id(tweets),
		"user_name": tools.get_user_name(tweets),
		"user_screen_name": tools.get_user_screen_name(tweets),
		"min_text_size": text_size_distro[0],
		"max_text_size": text_size_distro[1],
		"mean_text_size": text_size_distro[2],
		"median_text_size": text_size_distro[3],
		"std_text_size": text_size_distro[4],
		"skew_text_size": text_size_distro[5],
		"kurt_text_size": text_size_distro[6],
		"entropy_text_size": text_size_distro[7],
		"min_text_entropy": text_entropy_distributions[0],
		"max_text_entropy": text_entropy_distributions[1],
		"mean_text_entropy": text_entropy_distributions[2],
		"median_text_entropy": text_entropy_distributions[3],
		"std_text_entropy": text_entropy_distributions[4],
		"skew_text_entropy": text_entropy_distributions[5],
		"kurt_text_entropy": text_entropy_distributions[6],
		"entropy_text_entropy": text_entropy_distributions[7],
		"min_similarity": similarities_distro[0],
		"max_similarity": similarities_distro[1],
		"mean_similarity": similarities_distro[2],
		"median_similarity": similarities_distro[3],
		"std_similarity": similarities_distro[4],
		"skew_similarity": similarities_distro[5],
		"kurt_similarity": similarities_distro[6],
		"entropy_similarity": similarities_distro[7],
		"NN_proportion": pros_tag_proportions[0],
		"VB_proportion": pros_tag_proportions[1],
		"RB_proportion": pros_tag_proportions[2],
		"WP_proportion": pros_tag_proportions[3],
		"WDT_proportion": pros_tag_proportions[4],
		"DT_proportion": pros_tag_proportions[5],
		"JJ_proportion": pros_tag_proportions[6],
		"PRP_proportion": pros_tag_proportions[7],
		"UH_proportion": pros_tag_proportions[8],
		"min_NN": NN_distro[0],
		"max_NN": NN_distro[1],
		"mean_NN": NN_distro[2],
		"median_NN": NN_distro[3],
		"std_NN": NN_distro[4],
		"skew_NN": NN_distro[5],
		"kurt_NN": NN_distro[6],
		"entropy_NN": NN_distro[7],
		"min_VB": VB_distro[0],
		"max_VB": VB_distro[1],
		"mean_VB": VB_distro[2],
		"median_VB": VB_distro[3],
		"std_VB": VB_distro[4],
		"skew_VB": VB_distro[5],
		"kurt_VB": VB_distro[6],
		"entropy_VB": VB_distro[7],
		"min_RB": RB_distro[0],
		"max_RB": RB_distro[1],
		"mean_RB": RB_distro[2],
		"median_RB": RB_distro[3],
		"std_RB": RB_distro[4],
		"skew_RB": RB_distro[5],
		"kurt_RB": RB_distro[6],
		"entropy_RB": RB_distro[7],
		"min_WP": WP_distro[0],
		"max_WP": WP_distro[1],
		"mean_WP": WP_distro[2],
		"median_WP": WP_distro[3],
		"std_WP": WP_distro[4],
		"skew_WP": WP_distro[5],
		"kurt_WP": WP_distro[6],
		"entropy_WP": WP_distro[7],
		"min_DT": DT_distro[0],
		"max_DT": DT_distro[1],
		"mean_DT": DT_distro[2],
		"median_DT": DT_distro[3],
		"std_DT": DT_distro[4],
		"skew_DT": DT_distro[5],
		"kurt_DT": DT_distro[6],
		"entropy_DT": DT_distro[7],
		"min_WDT": WDT_distro[0],
		"max_WDT": WDT_distro[1],
		"mean_WDT": WDT_distro[2],
		"median_WDT": WDT_distro[3],
		"std_WDT": WDT_distro[4],
		"skew_WDT": WDT_distro[5],
		"kurt_WDT": WDT_distro[6],
		"entropy_WDT": WDT_distro[7],
		"min_JJ": JJ_distro[0],
		"max_JJ": JJ_distro[1],
		"mean_JJ": JJ_distro[2],
		"median_JJ": JJ_distro[3],
		"std_JJ": JJ_distro[4],
		"skew_JJ": JJ_distro[5],
		"kurt_JJ": JJ_distro[6],
		"entropy_JJ": JJ_distro[7],
		"min_PRP": PRP_distro[0],
		"max_PRP": PRP_distro[1],
		"mean_PRP": PRP_distro[2],
		"median_PRP": PRP_distro[3],
		"std_PRP": PRP_distro[4],
		"skew_PRP": PRP_distro[5],
		"kurt_PRP": PRP_distro[6],
		"entropy_PRP": PRP_distro[7],
		"min_UH": UH_distro[0],
		"max_UH": UH_distro[1],
		"mean_UH": UH_distro[2],
		"median_UH": UH_distro[3],
		"std_UH": UH_distro[4],
		"skew_UH": UH_distro[5],
		"kurt_UH": UH_distro[6],
		"entropy_UH": UH_distro[7],
		"max_appearance_of_punc_mark": cf.get_common_marks(tweets),
		"min_marks": marks_distro[0],
		"max_marks": marks_distro[1],
		"mean_marks": marks_distro[2],
		"median_marks": marks_distro[3],
		"std_marks": marks_distro[4],
		"skew_marks": marks_distro[5],
		"kurt_marks": marks_distro[6],
		"entropy_marks": marks_distro[7],
		"tweet_retweet_ratio": cf.get_tweet_retweet_ratio(tweets)[2],
		"min_tags": tags_distro[0],
		"max_tags": tags_distro[1],
		"mean_tags": tags_distro[2],
		"median_tags": tags_distro[3],
		"std_tags": tags_distro[4],
		"skew_tags": tags_distro[5],
		"kurt_tags": tags_distro[6],
		"entropy_tags": tags_distro[7],
		"min_urls": urls_distro[0],
		"max_urls": urls_distro[1],
		"mean_urls": urls_distro[2],
		"median_urls": urls_distro[3],
		"std_urls": urls_distro[4],
		"skew_urls": urls_distro[5],
		"kurt_urls": urls_distro[6],
		"entropy_urls": urls_distro[7],
		"min_mentions": mentions_distro[0],
		"max_mentions": mentions_distro[1],
		"mean_mentions": mentions_distro[2],
		"median_mentions": mentions_distro[3],
		"std_mentions": mentions_distro[4],
		"skew_mentions": mentions_distro[5],
		"kurt_mentions": mentions_distro[6],
		"entropy_mentions": mentions_distro[7],
		"min_symbols": symbols_distro[0],
		"max_symbols": symbols_distro[1],
		"mean_symbols": symbols_distro[2],
		"median_symbols": symbols_distro[3],
		"std_symbols": symbols_distro[4],
		"skew_symbols": symbols_distro[5],
		"kurt_symbols": symbols_distro[6],
		"entropy_symbols": symbols_distro[7],
		"min_media": media_distro[0],
		"max_media": media_distro[1],
		"mean_media": media_distro[2],
		"median_media": media_distro[3],
		"std_media": media_distro[4],
		"skew_media": media_distro[5],
		"kurt_media": media_distro[6],
		"entropy_media": media_distro[7],
		"source_change": cf.source_change(tweets),
		"source_types": cf.source_types(tweets),
		"unique_mentions_rate": cf.get_unique_mentions_rate(tweets),
		"min_favs": favs_distro[0],
		"max_favs": favs_distro[1],
		"mean_favs": favs_distro[2],
		"median_favs": favs_distro[3],
		"std_favs": favs_distro[4],
		"skew_favs": favs_distro[5],
		"kurt_favs": favs_distro[6],
		"entropy_favs": favs_distro[7],
		"min_rts": rts_distro[0],
		"max_rts": rts_distro[1],
		"mean_rts": rts_distro[2],
		"median_rts": rts_distro[3],
		"std_rts": rts_distro[4],
		"skew_rts": rts_distro[5],
		"kurt_rts": rts_distro[6],
		"entropy_rts": rts_distro[7],
		"min_tokens": text_distro[0],
		"max_tokens": text_distro[1],
		"mean_tokens": text_distro[2],
		"median_tokens": text_distro[3],
		"std_tokens": text_distro[4],
		"skew_tokens": text_distro[5],
		"kurt_tokens": text_distro[6],
		"entropy_tokens": text_distro[7]
	}

	return content_vector

#APPLIES TO ALL ACCOUNTS - IF NO TWEETS/NO RETWEETS TAKES INTO CONSIDERATION WHATEVER TYPE OF CONTENT. OTHERWISE ONLY TWEETS
def get_sentiment_feature_vector(tweets):
	only_tweets = tools.get_only_tweets(tweets)
	retweets = tools.get_retweets(tweets)
	if len(only_tweets) == 0:
		only_tweets = tweets
	emojis_per_tweet_ratio_distros = sf.get_emojis_per_tweet(only_tweets)
	pos_emoji_distro = sf.get_positive_negative_neutral_emojis_per_tweet(only_tweets)[2]
	neg_emoji_distro = sf.get_positive_negative_neutral_emojis_per_tweet(only_tweets)[1]
	neu_emoji_distro = sf.get_positive_negative_neutral_emojis_per_tweet(only_tweets)[0]
	pos_sentiment_distro = sf.get_positive_sentiment_per_tweet(only_tweets)
	neg_sentiment_distro = sf.get_negative_sentiment_per_tweet(only_tweets)
	neu_sentiment_distro = sf.get_neutral_sentiment_per_tweet(only_tweets)
	sentiment_vector={
		"user_id": tools.get_user_id(tweets),
		"user_name": tools.get_user_name(tweets),
		"user_screen_name": tools.get_user_screen_name(tweets),
		"tweet_emoji_ratio" : sf.tweet_emoji_ratio(only_tweets),
		"most_common_emoji" : sf.get_most_common_emoji(only_tweets),
		"min_emoji_per_tweet": emojis_per_tweet_ratio_distros[0],
		"max_emoji_per_tweet": emojis_per_tweet_ratio_distros[1],
		"mean_emoji_per_tweet": emojis_per_tweet_ratio_distros[2],
		"median_emoji_per_tweet": emojis_per_tweet_ratio_distros[3],
		"std_emoji_per_tweet": emojis_per_tweet_ratio_distros[4],
		"skew_emoji_per_tweet": emojis_per_tweet_ratio_distros[5],
		"kurt_emoji_per_tweet": emojis_per_tweet_ratio_distros[6],
		"entropy_emoji_per_tweet": emojis_per_tweet_ratio_distros[7],
		"min_pos_emoji_per_tweet": pos_emoji_distro[0],
		"max_pos_emoji_per_tweet": pos_emoji_distro[1],
		"mean_pos_emoji_per_tweet": pos_emoji_distro[2],
		"median_pos_emoji_per_tweet": pos_emoji_distro[3],
		"std_pos_emoji_per_tweet": pos_emoji_distro[4],
		"skew_pos_emoji_per_tweet": pos_emoji_distro[5],
		"kurt_pos_emoji_per_tweet": pos_emoji_distro[6],
		"entropy_pos_emoji_per_tweet": pos_emoji_distro[7],
		"min_neg_emoji_per_tweet": neg_emoji_distro[0],
		"max_neg_emoji_per_tweet": neg_emoji_distro[1],
		"mean_neg_emoji_per_tweet": neg_emoji_distro[2],
		"median_neg_emoji_per_tweet": neg_emoji_distro[3],
		"std_neg_emoji_per_tweet": neg_emoji_distro[4],
		"skew_neg_emoji_per_tweet": neg_emoji_distro[5],
		"kurt_neg_emoji_per_tweet": neg_emoji_distro[6],
		"entropy_neg_emoji_per_tweet": neg_emoji_distro[7],
		"min_neu_emoji_per_tweet": neu_emoji_distro[0],
		"max_neu_emoji_per_tweet": neu_emoji_distro[1],
		"mean_neu_emoji_per_tweet": neu_emoji_distro[2],
		"median_neu_emoji_per_tweet": neu_emoji_distro[3],
		"std_neu_emoji_per_tweet": neu_emoji_distro[4],
		"skew_neu_emoji_per_tweet": neu_emoji_distro[5],
		"kurt_neu_emoji_per_tweet": neu_emoji_distro[6],
		"entropy_neu_emoji_per_tweet": neu_emoji_distro[7],
		"min_pos_sent_per_tweet": pos_sentiment_distro[0],
		"max_pos_sent_per_tweet": pos_sentiment_distro[1],
		"mean_pos_sent_per_tweet": pos_sentiment_distro[2],
		"median_pos_sent_per_tweet": pos_sentiment_distro[3],
		"std_pos_sent_per_tweet": pos_sentiment_distro[4],
		"skew_pos_sent_per_tweet": pos_sentiment_distro[5],
		"kurt_pos_sent_per_tweet": pos_sentiment_distro[6],
		"entropy_pos_sent_per_tweet": pos_sentiment_distro[7],
		"min_neg_sent_per_tweet": neg_sentiment_distro[0],
		"max_neg_sent_per_tweet": neg_sentiment_distro[1],
		"mean_neg_sent_per_tweet": neg_sentiment_distro[2],
		"median_neg_sent_per_tweet": neg_sentiment_distro[3],
		"std_neg_sent_per_tweet": neg_sentiment_distro[4],
		"skew_neg_sent_per_tweet": neg_sentiment_distro[5],
		"kurt_neg_sent_per_tweet": neg_sentiment_distro[6],
		"entropy_neg_sent_per_tweet": neg_sentiment_distro[7],
		"min_neu_sent_per_tweet": neu_sentiment_distro[0],
		"max_neu_sent_per_tweet": neu_sentiment_distro[1],
		"mean_neu_sent_per_tweet": neu_sentiment_distro[2],
		"median_neu_sent_per_tweet": neu_sentiment_distro[3],
		"std_neu_sent_per_tweet": neu_sentiment_distro[4],
		"skew_neu_sent_per_tweet": neu_sentiment_distro[5],
		"kurt_neu_sent_per_tweet": neu_sentiment_distro[6],
		"entropy_neu_sent_per_tweet": neu_sentiment_distro[7]
	}

	return sentiment_vector

#CAN BE USED FOR ALL ACCOUNTS - DESPITE THE EXISTENCE OF TWEETS/RETWEETS#
def get_network_feature_vector(tweets):
	G,weights = nf.get_hashtag_network(tweets)
	weights_distro = nf.get_statistical_results_of_list(weights)
	graph_features = nf.graph_features(G)
	network_vector={
		"user_id": tools.get_user_id(tweets),
		"user_name": tools.get_user_name(tweets),
		"user_screen_name": tools.get_user_screen_name(tweets),
		"density" : graph_features[0],
		"avg_clustering" : graph_features[1],
		"triangles" : len(graph_features[2]),
		"volume" : graph_features[3],
		"mass" : graph_features[4],
		"min_weight": weights_distro[0],
		"max_max_weight": weights_distro[1],
		"mean_weight": weights_distro[2],
		"median_weight": weights_distro[3],
		"std_weight": weights_distro[4],
		"skew_weight": weights_distro[5],
		"kurt_weight": weights_distro[6],
		"entropy_weight": weights_distro[7]
	}
	lista = list(network_vector.keys())
	pickle.dump(lista, open('network_features', 'wb'))
	return network_vector

def get_all_features_train(user,tweets,labels):
	user_vector = get_user_vector(user)
	friend = get_friend_feature_vector(tweets)
	temporal = get_temporal_features_vector(tweets)
	content = get_content_feature_vector(tweets)
	sentiment = get_sentiment_feature_vector(tweets)
	network = get_network_feature_vector(tweets)
	total = {**user_vector, **friend, **temporal, **content, **sentiment, **network}
	total['label'] = labels[user['id_str']]
	return total

def get_all_features_no_rts_train(user,tweets,labels):
	user_vector = get_user_vector(user)
	temporal = get_temporal_features_vector_only_twts(tweets)
	content = get_content_feature_vector_twts(tweets)
	sentiment = get_sentiment_feature_vector(tweets)
	network = get_network_feature_vector(tweets)
	total = {**user_vector,**temporal, **content, **sentiment, **network}
	total['label'] = labels[user['id_str']]
	return total

def get_all_features(tweets):
	user = tools.get_user_object(tweets)
	user_vector = get_user_vector(user)
	friend = get_friend_feature_vector(tweets)
	temporal = get_temporal_features_vector(tweets)
	content = get_content_feature_vector(tweets)
	sentiment = get_sentiment_feature_vector(tweets)
	network = get_network_feature_vector(tweets)
	total = {**user_vector, **friend, **temporal, **content, **sentiment, **network}
	return total

def get_all_features_no_rts(user,tweets):
	user_vector = get_user_vector(user)
	temporal = get_temporal_features_vector_only_twts(tweets)
	content = get_content_feature_vector_twts(tweets)
	sentiment = get_sentiment_feature_vector(tweets)
	network = get_network_feature_vector(tweets)
	total = {**user_vector,**temporal, **content, **sentiment, **network}
	return total

def get_user_features(user,label):
	user_vector = get_user_vector(user)
	total = {**user_vector}
	total['label']= label[user['id_str']]
	return total