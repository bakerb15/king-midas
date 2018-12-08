from brain import Brain
from market import Market
from agent import Agent
import datetime
import pandas
import numpy as np
import time
from collections import OrderedDict


def create_dataset(dataset, forward=0, look_back=1):
  dataX, dataY = [], []
  for i in range(len(dataset)-look_back-forward+1):
    a = dataset[i:(i+look_back), :]
    dataX.append(a)
    dataY.append(dataset[i + forward+look_back - 1, :])
  return np.array(dataX), np.array(dataY)



HISTORY = 30 # how many days to use for a prediction
FUTURE = 15 # number of trading days to predict in the future
COMPANIES = 3

path_to_sp500 = './data/^GSPC.csv'
path_to_comp1 = './data/BA.csv'
path_to_comp2 = './data/BP.csv'
path_to_comp3 = './data/CAT.csv'
sp = pandas.read_csv(path_to_sp500, index_col=0)
ba = pandas.read_csv(path_to_comp1, index_col=0)
bp = pandas.read_csv(path_to_comp2, index_col=0)
cat = pandas.read_csv(path_to_comp3, index_col=0)

entities = [('^GSPC',sp) ,('BA', ba),('BP', bp), ('CAT', cat)]

columns = ['Open','High','Low', 'Close','AdjClose','Volume']

raw_dataframe = OrderedDict()

for symbol, data in entities:
    for col in columns:
        raw_dataframe['{}_{}'.format(symbol, col)] = data[col].values

stock_market = pandas.DataFrame(raw_dataframe).values



split = np.array_split(stock_market, len(stock_market))


train_x, train_y = create_dataset(stock_market, forward=FUTURE, look_back=HISTORY)
# train_x = []
# train_y = []
# bufferx =[]
# buffery =[]
# for frame in split:
#     if len(bufferx) == INPUT_LENGTH:
#         buffery.append(frame)
#         train_x.append(bufferx)
#         train_y.append(buffery)
#         bufferx = []
#         buffery = []
#         bufferx.append(frame)
#     else:
#         bufferx.append(frame)




b = Brain(HISTORY, 24)



start_train = time.process_time()
b.model.fit(train_x, train_y, epochs=10)
end_train = time.process_time()
print('training time: {}'.format(end_train - start_train))

result = b.model.predict(train_x)
print('END')
# TRANSACTION_COST = 10.0
#print('Creating market')
# mrk = Market(TRANSACTION_COST)
#print('Market data loaded')
#whole_start = datetime.date(2009, 11, 3)
#whole_end = datetime.date(2012, 12, 12)


