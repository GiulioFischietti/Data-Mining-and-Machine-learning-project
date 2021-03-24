import time
import pandas as pd
import numpy as np

results = pd.read_csv("./Tweets/results/Biden/results.csv", sep=",", encoding = 'latin1', error_bad_lines=False)
# print(results.info())




# splittare il dataset per eventi

trump_peaks = ["2020-09-30", "2020-10-05", "2020-10-12", "2020-11-07", "2020-11-16", "2020-12-12", "2021-01-06", "2021-01-08"]
trump_peaks_split = ["2020-09-01", "2020-10-01", "2020-10-06", "2020-10-13", "2020-11-08", "2020-11-16", "2020-12-13", "2021-01-07"]

# peak dates
biden_peaks = ["2020-09-30", "2020-10-06", "2020-10-12", "2020-10-23", "2020-11-04", "2020-11-25", "2020-12-20", "2021-01-07", "2021-01-08"]

# dates increased by one so that I can train again the model after the total occurrence of the event
biden_peaks_split = ["2020-09-01", "2020-10-01", "2020-10-07", "2020-10-13", "2020-10-24", "2020-11-05", "2020-11-26", "2020-12-21", "2021-01-08"]



results['date'] = results['date'].astype('datetime64[ns]')
results = results.set_index(results['date'])
results = results.sort_index()

dataframe_split = []




for index, peak in enumerate(biden_peaks_split):
    if((index) < len(biden_peaks_split)):
        dataframe_split.append(results[peak: biden_peaks[index]])

counter = 0
for index, split in enumerate(dataframe_split):
    userdata = []
    stancedata = []
    
    print(len(split['user_id'].unique()))

    start_time = time.time()
    for user, df_user in split.groupby(['user_id']):
        counter += 1
        
        total = len(df_user['stance'])
        positive = np.count_nonzero(df_user['stance'] == 1)
        negative = total-positive

        # if(counter % 1000 == 0):
        #     print(str(positive + negative) + ' ' + str(positive) + ' ' + str(negative))
        #     elapsed_time = time.time() - start_time
        #     print(elapsed_time)
        user_score = (positive - negative)/(total)
        # print(user_score)
        userdata.append(user)
        if(user_score >= 0.5):
            stancedata.append(1)
            
            # data.append(pd.DataFrame(data=[{"user_id": user}, {"stance": "1"}], columns=['user_id', 'stance']),ignore_index=True)
        else: 
            if(user_score <= -0.5):
                stancedata.append(0)
                # data.append(pd.DataFrame(data=[{"user_id": user}, {"stance": "0"}], columns=['user_id', 'stance']),ignore_index=True)
            else: 
                stancedata.append(2)
                # data.append(pd.DataFrame(data=[{"user_id": user}, {"stance": "2"}], columns=['user_id', 'stance']),ignore_index=True)
    
    df_user_stance = pd.DataFrame(data={"user_id": userdata, "stance": stancedata, "event": index})
    df_user_stance.to_csv("./Tweets/stance/Biden/stance of users until " + biden_peaks[index] + ".csv", index=False)
    print(df_user_stance.info())


# per ogni split:
#   generare un dizionario (dataframe) che contiene user_id unici relativi al periodo
#   generare un dataframe che conterrà i risultati
#   per ogni user_id:
#       calcolare la sua stance con la formula DM e aggiungere il risultato al dataframe
#   stampare il risultato in un csv con nome unico

