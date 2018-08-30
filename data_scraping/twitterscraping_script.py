import pandas as pd
import numpy as np
import os
from twitterscraper import query_tweets
from twitterscraper.query import query_tweets_from_user

username = "@Gil_Hoffman"
c.Output = r"data_scraping/data_retweet/test_csv.csv"
try:
    os.remove("data_scraping/data_retweet/test_csv.csv")
except:
    pass

# CSV Fieldnames
list_of_tweets = query_tweets_from_user(username, limit=10)

#print the retrieved tweets to the screen:
for tweet in list_of_tweets:
    print(tweet.from_soup())

#Or save the retrieved tweets to file:
# with open("data_scraping/data_retweet/test.txt",'w') as file:
#     for tweet in query_tweets("Trump OR Clinton", 10):
#         file.write(tweet.encode('utf-8'))
# file.close()

if __name__ == '__main__':
    # recieve list of participant id in seed group
    pro_colname = "Pro-Israeli sources on Twitter"
    anti_colname = "Anti Israeli sources on Twitter"
    seed_list = pd.read_csv(r"data_scraping/twitter_seeds.csv", header=0)

    for col in [pro_colname, anti_colname]:
        for account in seed_list[col]:
            print("@"+account.split('/')[-1])
            username = account.split('/')
            c.Username = "@"+username
            c.Output = "data/{username}.csv"

    # list_of_tweets = query_tweets("Trump OR Clinton", 10)
    #
    # #print the retrieved tweets to the screen:
    # for tweet in query_tweets("Trump OR Clinton", 10):
    #     print(tweet)
    #
    # #Or save the retrieved tweets to file:
    # # file = open(“output.txt”,”w”)
    # for tweet in query_tweets("Trump OR Clinton", 10):
    #     file.write(tweet.encode('utf-8'))
    # file.close()
