import pandas as pd
import numpy as np
import os
import twint

c = twint.Config()
c.Username = "@Gil_Hoffman"
c.Store_csv = True
c.Limit = 10
c.Output = r"data_scraping/data_retweet/test_csv.csv"
try:
    os.remove("data_scraping/data_retweet/test_csv.csv")
except:
    pass
# CSV Fieldnames
c.Custom = ["id", "username", "retweets","user_rt","retweet"]
twint.run.Search(c)

if __name__ == '__main__':
    # recieve list of participant id in seed group
    pro_colname = "Pro-Israeli sources on Twitter"
    anti_colname = "Anti Israeli sources on Twitter"
    seed_list = pd.read_csv(r"data_scraping/twitter_seeds.csv", header=0)

    c = twint.Config()
    c.Username = "@Gil_Hoffman"
    c.Store_csv = True
    c.Output = "AvivaKlompas.csv"
    try:
        os.remove("AvivaKlompas.csv")
    except:
        pass
    twint.run.Followers(c)

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
