from brain import Brain
from market import Market
from agent import Agent
import datetime
import pandas
import numpy as np
import time
from collections import OrderedDict

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
        raw_dataframe['{}_{}'.format(symbol, col)] = data[col]

stock_market = pandas.DataFrame(raw_dataframe)

split = np.array_split(stock_market, 10)


print(stock_market)

b = Brain(24)

train_x, train_y = split[0][:-1], split[0][1:]

start_train = time.process_time()
b.model.fit(train_x, train_y, epochs=10)
end_train = time.process_time()
print('training time: {}'.format(end_train - start_train))


print('END')
# TRANSACTION_COST = 10.0
#print('Creating market')
# mrk = Market(TRANSACTION_COST)
#print('Market data loaded')
#whole_start = datetime.date(2009, 11, 3)
#whole_end = datetime.date(2012, 12, 12)


