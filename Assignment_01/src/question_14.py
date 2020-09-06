import pandas as pd
import numpy as np

train_df = pd.read_csv("include/train.csv")
test_df = pd.read_csv("include/test.csv")


unique_ticket = train_df['Ticket'].unique()
# print(unique_ticket)
print("Total number of ticket entries ", len(train_df['Ticket']))
print("Number of unique ticket values ", len(unique_ticket))

# rate of duplicates = (total records - unique records)/total records

Ticket_dup_rate = (len(train_df['Ticket']) - len(unique_ticket)) * 100 /len(train_df['Ticket'])
print("Total ticket duplicate rate is ", Ticket_dup_rate, "%")