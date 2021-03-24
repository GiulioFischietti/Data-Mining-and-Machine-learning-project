import pandas as pd
import numpy as np

results = pd.read_csv("./Tweets/results/Trump/results.csv", sep=",", encoding = 'latin1', error_bad_lines=False)
results['date'] = results['date'].astype('datetime64[ns]')


#defined as favor/not in favor
stanceratio = []
stancedate = []

for date, df_date in results.groupby(['date']):
    total = len(df_date['stance'])
    positive = np.count_nonzero(df_date['stance'] == 1)
    negative = total-positive
    
    stanceratio.append(positive/negative)
    stancedate.append(date)

df_stanceratio = pd.DataFrame(data={"stance": stanceratio, "date": stancedate})
df_stanceratio.sort_values(by=['stance'], inplace=True)

print(df_stanceratio.info())
print(df_stanceratio.iloc[[-1, -2, -3, -4, -5]])
print(df_stanceratio.iloc[[0, 1, 2, 3, 4]])