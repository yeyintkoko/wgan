import pandas as pd
import datetime

def parser(x):
    return datetime.datetime.strptime(x, '%m/%d/%Y')

# Read the data from the CSV
aapl = pd.read_csv("data/AAPL.csv", header=0, index_col=0, parse_dates=[0], date_format=parser)

# Use the index instead of a column named 'Date'
data = pd.DataFrame(index=aapl.index)  # Create a new DataFrame with the same index as aapl
data['Date'] = aapl['Date']
data['price'] = aapl['Close/Last']

# Save the combined data to a CSV
data.to_csv('data/panel_data_close.csv')
