from brain import Brain
from market import Market
from agent import Agent
import datetime
import pandas
import numpy as np
import time
import os
from market_data import market_index, company
from collections import OrderedDict, deque
from dataprep import create_dataset

# the stocks with largest increase by percentage
def select_a( evidence , target, howmany):
    howmany_hot = []
    prepro = []
    buffer = deque()
    last = evidence[-1]
    for i in range(len(target) - 1):
        if i % 6 == 4:
            close0 = last[i]
            close1 = target[i]
            if close0 > 0:
                prc = (close1 - close0)/close0
                prepro.append(prc)
            else:
                prepro.append(0.0)
    for i in range(len(prepro)):
        if len(buffer) < howmany:
            buffer.append((i, prepro[i]))
        else:
            if prepro[i] > buffer[:-1][1]:
                buffer.appendleft((i, prepro[i]))
                while len(buffer) > howmany:
                    buffer.pop()
        sorted(buffer, reverse=True)
    selected_indices = {}
    for item in buffer:
        selected_indices[item[0]] = True

    for i in len(int(target/6)):
        if i in selected_indices:
            howmany_hot.append(1)
        else:
            howmany_hot.append(0)

    return howmany_hot


HISTORY = 20         # how many days to use for a prediction
FUTURE = 20           # number of trading days to predict in the future
HOW_MANY_TO_PICK = 8 # number of stocks to pick


# path_to_sp500 = './data/^GSPC.csv'
# path_to_comp1 = './data/BA.csv'
# path_to_comp2 = './data/BP.csv'
# path_to_comp3 = './data/CAT.csv'
# sp = pandas.read_csv(path_to_sp500, index_col=0)
# ba = pandas.read_csv(path_to_comp1, index_col=0)
# bp = pandas.read_csv(path_to_comp2, index_col=0)
# cat = pandas.read_csv(path_to_comp3, index_col=0)
#
# entities = [('^GSPC',sp) ,('BA', ba),('BP', bp), ('CAT', cat)]

entities = OrderedDict()
for symbol in company.values():
    entities[symbol] = pandas.read_csv(os.path.join('./data/', symbol + '.csv'))
for symbol in market_index.values():
    entities[symbol] = pandas.read_csv(os.path.join('./data/', symbol + '.csv'))


columns = ['Open','High','Low', 'Close','AdjClose','Volume']

raw_full = OrderedDict()
raw_company_only = OrderedDict()

for symbol in entities:
    if symbol in company.values():
        for col in columns:
            raw_company_only['{}_{}'.format(symbol, col)] = entities[symbol][col].values
    for col in columns:
        raw_full['{}_{}'.format(symbol, col)] = entities[symbol][col].values


full = pandas.DataFrame(raw_full).values
company_only = pandas.DataFrame(raw_company_only).values
#stock_markeraw_fullt = pandas.DataFrame(raw_dataframe).values



train_x, train_y = create_dataset(full, company_only, select_a, HOW_MANY_TO_PICK, forward=FUTURE, look_back=HISTORY)


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
print(result)
print('END')
# TRANSACTION_COST = 10.0
#print('Creating market')
# mrk = Market(TRANSACTION_COST)
#print('Market data loaded')
#whole_start = datetime.date(2009, 11, 3)
#whole_end = datetime.date(2012, 12, 12)


