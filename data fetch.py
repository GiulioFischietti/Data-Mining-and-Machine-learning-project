# FETCH TWEETS 

import twint

c = twint.Config()
c.Search = "since:2021-01-29 until:2021-01-28 filter:replies -filter:links -filter:retweets"
c.Limit = 2000000
c.Lang = 'en'
c.Store_csv = True
c.Username  = "joeBiden"
c.Custom["tweet"] = ["user_id", "id", "conversation_id", "date", "tweet", "hashtags" ]

c.Hide_output = True
c.Output = "tweets test.csv"
c.Resume = "my_search_idboh_.txt"
# Run
twint.run.Search(c)




# CLEANING AND FILTERING TWEETS, REMOVING OTHER LANGUAGES TWEETS, EMOJIS 

# import csv
# import html
# from langdetect import detect

# csv_file = open('tweets to joe biden.csv', encoding='utf-8')
# csv_reader = csv.reader(csv_file, delimiter=',')

# languages = []

# with open("tweets to joe biden preremoval.csv", "w", encoding="utf-8", newline="") as file:
#     csv_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#     for row in csv_reader:
#         try:
#             csv_writer.writerow([row[0], row[1], row[2], row[3], html.unescape((row[4]).encode('ascii', 'ignore').decode('utf8')), row[5]])
                
#         except:
#             print('errore')