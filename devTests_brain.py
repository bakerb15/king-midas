from brain import Brain
from market import Market
from agent import Agent
import datetime
import pandas
import numpy as np
import time
import os
from sklearn.model_selection import train_test_split
from market_data import market_index, company
from collections import OrderedDict, deque
from dataprep import create_dataset

# the stocks with largest increase by percentage
def select_a( evidence , target, howmany, attributes_per_company):
    howmany_hot = []
    prepro = []
    buffer = deque()
    last = evidence[-1]
    for i in range(len(target)):
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
            buffer = deque(sorted(buffer, key=lambda x: x[1], reverse=True))
            if prepro[i] > buffer[-1][1]:
                buffer.appendleft((i, prepro[i]))
                while len(buffer) > howmany:
                    buffer.pop()

    selected_indices = {}
    for item in buffer:
        selected_indices[item[0]] = True

    for i in range(int(len(target)/attributes_per_company)):
        if i in selected_indices:
            howmany_hot.append(1.0
        else:
            howmany_hot.append(0.0)

    return howmany_hot

# the stocks with largest increase by percentage
def select_b( evidence , target, howmany, attributes_per_company):
    howmany_hot = []
    prepro = []
    buffer = deque()
    last = evidence[-1]
    for i in range(len(target)):
        if i % 6 == 4:
            close0 = last[i]
            close1 = target[i]
            prc = (close1 - close0)/close0
            prepro.append(prc)
    return prepro

CURRENT_DATE = pandas.datetime(2016, 11, 28)
HISTORY = 100       # how many days to use for a prediction
FUTURE = 30          # number of trading days to predict in the future
HOW_MANY_TO_PICK = 10 # number of stocks to pick. Unused for select_b
ATTR_PER_COMPANY = 6  #number of attributes per company and market index
COMPANIES_AND_INDICES = 0 # set when data is read in
COMPANIES = 0 # set when datas read
INPUT_VECTOR_DIM = 0 # set after data is read


entities = OrderedDict()
for symbol in company.values():
    entities[symbol] = pandas.read_csv(os.path.join('./data/', symbol + '.csv'))
    entities[symbol]['Date'] = pandas.to_datetime(entities[symbol]['Date'], format='%Y-%m-%d')
    mask = (entities[symbol]['Date'] < CURRENT_DATE)
    entities[symbol] = entities[symbol].loc[mask]
    COMPANIES_AND_INDICES += 1
    COMPANIES += 1
for symbol in market_index.values():
    entities[symbol] = pandas.read_csv(os.path.join('./data/', symbol + '.csv'))
    entities[symbol]['Date'] = pandas.to_datetime(entities[symbol]['Date'], format='%Y-%m-%d')
    mask = (entities[symbol]['Date'] < CURRENT_DATE)
    entities[symbol] = entities[symbol].loc[mask]
    COMPANIES_AND_INDICES += 1
INPUT_VECTOR_DIM = COMPANIES_AND_INDICES * ATTR_PER_COMPANY


columns = ['Open','High','Low', 'Close','AdjClose','Volume']

raw_full = OrderedDict()
raw_company_only = OrderedDict()

for symbol in entities:
    if symbol in company.values():
        for col in columns:
            raw_company_only['{}_{}'.format(symbol, col)] = entities[symbol][col]
    for col in columns:
        raw_full['{}_{}'.format(symbol, col)] = entities[symbol][col]




full = pandas.DataFrame(raw_full).values
company_only = pandas.DataFrame(raw_company_only).values

X, Y = create_dataset(full, company_only, select_a, HOW_MANY_TO_PICK, ATTR_PER_COMPANY, forward=FUTURE, look_back=HISTORY)


b = Brain(HISTORY, INPUT_VECTOR_DIM, COMPANIES)

x_train, x_test, y_train, y_test =  train_test_split(X, Y, test_size=0.30)


start_train = time.process_time()
b.model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=1, epochs=30)
end_train = time.process_time()
print('training and validation time: {}'.format(end_train - start_train))

model_json = b.model.to_json()
with open("model_b.json", "w") as json_file:
    json_file.write(model_json)
b.model.save_weights('model_b.h5')

print('END')



