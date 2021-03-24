from os import error
import pandas as pd

file_name = "./Tweets/original/Tweets to Trump/tweets to trump part 5 decoded.csv"
file_name_output = "./Tweets/original/Tweets to Trump/tweets to trump part 5 decoded without duplicates.csv"

df = pd.read_csv(file_name, sep=",", encoding = 'utf_8', error_bad_lines=False)

# Notes:
# - the `subset=None` means that every column is used 
#    to determine if two rows are different; to change that specify
#    the columns as an array
# - the `inplace=True` means that the data structure is changed and
#   the duplicate rows are gone  
df.drop_duplicates(subset=None, inplace=True)

# Write the results to a different file
df.to_csv(file_name_output, index=False)