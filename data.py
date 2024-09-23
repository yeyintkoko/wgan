import pandas as pd
import datetime

def parser(x):
    return datetime.datetime.strptime(x, '%m/%d/%Y')

# Read the data from the CSV
aapl = pd.read_csv("data/HistoricalData_AAPL.csv", header=0, index_col=0, parse_dates=[0], date_format=parser)
bac = pd.read_csv("data/HistoricalData_BAC.csv", header=0, index_col=0, parse_dates=[0], date_format=parser)
amd = pd.read_csv("data/HistoricalData_AMD.csv", header=0, index_col=0, parse_dates=[0], date_format=parser)
meta = pd.read_csv("data/HistoricalData_META.csv", header=0, index_col=0, parse_dates=[0], date_format=parser)
msft = pd.read_csv("data/HistoricalData_MSFT.csv", header=0, index_col=0, parse_dates=[0], date_format=parser)
spx = pd.read_csv("data/HistoricalData_SPX.csv", header=0, index_col=0, parse_dates=[0], date_format=parser)
crm = pd.read_csv("data/HistoricalData_CRM.csv", header=0, index_col=0, parse_dates=[0], date_format=parser)

# Use the index instead of a column named 'Date'
data = pd.DataFrame(index=aapl.index)  # Create a new DataFrame with the same index as aapl
data['price'] = aapl['Close/Last']
data['bac'] = bac['Close/Last']
data['meta'] = meta['Close/Last']
data['msft'] = msft['Close/Last']
data['spx'] = spx['Close/Last']
data['amd'] = amd['Close/Last']
data['crm'] = crm['Close/Last']

# Save the combined data to a CSV
data.to_csv('data/panel_data_close.csv')
