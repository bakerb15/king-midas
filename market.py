from market_data import company, market_index
import os
from enum import Enum
import pandas
import datetime
from collections import OrderedDict
from agent import Agent
from keras.models import model_from_json
from dataprep import create_dataset, create_dataset2
import numpy as np
import random

class Market(object):

    def __init__(self, transaction_cost):
        self.transaction_cost = transaction_cost
        self.input_data = None
        self.truth_data = None
        self.current_day = None
        self.simulation_in_progress = False
        self.end_date = None
        self.input_for_agents = None
        self.agents = []
        self.step_count = 0

    def create_simulation(self, start_date, end_date):
        entities = OrderedDict()
        for symbol in company.values():
            entities[symbol] = pandas.read_csv(os.path.join('./data/', symbol + '.csv'))
            entities[symbol]['Date'] = pandas.to_datetime(entities[symbol]['Date'], format='%Y-%m-%d')
            mask = (entities[symbol]['Date'] > start_date)
            entities[symbol] = entities[symbol].loc[mask]
        for symbol in market_index.values():
            entities[symbol] = pandas.read_csv(os.path.join('./data/', symbol + '.csv'))
            entities[symbol]['Date'] = pandas.to_datetime(entities[symbol]['Date'], format='%Y-%m-%d')
            mask = (entities[symbol]['Date'] > start_date)
            entities[symbol] = entities[symbol].loc[mask]

        columns = ['Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume']

        raw_full = OrderedDict()

        for symbol in entities:
            for col in columns:
                raw_full['{}_{}'.format(symbol, col)] = entities[symbol][col]



        self.truth_data = entities
        self.input_data = pandas.DataFrame(raw_full).values
        self.current_day = start_date
        self.end_date = end_date
        self.simulation_in_progress = True

        HISTORY = 100  # how many days to use for a prediction
        FUTURE = 30  # number of trading days to predict in the future
        HOW_MANY_TO_PICK = 10  # number of stocks to pick. Unused for select_b
        ATTR_PER_COMPANY = 6  # number of attributes per company and market index
        company_only = None
        select_a = None
        full = self.input_data
        self.input_for_agents = create_dataset(full, company_only, select_a, HOW_MANY_TO_PICK, ATTR_PER_COMPANY, forward=FUTURE, look_back=HISTORY, only_x=True)
        self.current_index = 0

    def add_me(self, agent):
        self.agents.append(agent)

    def create_simulation_SandPonly(self, start_date, end_date):
        entities = OrderedDict()
        # for symbol in company.values():
        #     entities[symbol] = pandas.read_csv(os.path.join('./data/', symbol + '.csv'))
        #     entities[symbol]['Date'] = pandas.to_datetime(entities[symbol]['Date'], format='%Y-%m-%d')
        #     mask = (entities[symbol]['Date'] > start_date)
        #     entities[symbol] = entities[symbol].loc[mask]
        for symbol in market_index.values():
            if symbol == '^GSPC':
                entities[symbol] = pandas.read_csv(os.path.join('./data/', symbol + '.csv'))
                entities[symbol]['Date'] = pandas.to_datetime(entities[symbol]['Date'], format='%Y-%m-%d')
                mask = (entities[symbol]['Date'] > start_date)
                entities[symbol] = entities[symbol].loc[mask]

        columns = ['Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume']

        raw_full = OrderedDict()

        for symbol in entities:
            for col in columns:
                raw_full['{}_{}'.format(symbol, col)] = entities[symbol][col]



        self.truth_data = entities
        self.input_data = pandas.DataFrame(raw_full).values
        self.current_day = start_date
        self.end_date = end_date
        self.simulation_in_progress = True

        HISTORY = 100  # how many days to use for a prediction
        FUTURE = 30  # number of trading days to predict in the future
        HOW_MANY_TO_PICK = 10  # number of stocks to pick. Unused for select_b
        ATTR_PER_COMPANY = 6  # number of attributes per company and market index
        company_only = None
        select_only1 = None
        full = self.input_data
        self.input_for_agents = create_dataset(full, company_only, select_only1, HOW_MANY_TO_PICK, ATTR_PER_COMPANY, forward=FUTURE, look_back=HISTORY, only_x=True)
        self.current_index = 0


    def add_agent(self, agent):
        self.agents.append(agent)

    def step(self):
        self.step_count += 1
        info = self.get_current_data()
        if info is None:
            return False
        else:
            for agent in self.agents:
                agent.observe(info)
        return True

    def get_current_data(self):
        if self.current_index < len(self.input_for_agents):
            ret = np.array([self.input_for_agents[self.current_index]])
            self.current_index += 1
            return ret
        else:
            return None

    def value_of_shares(self, shares):
        current_prices = self.input_data[self.current_index]
        high = current_prices[2]
        low =  current_prices[3]
        value = random.uniform(low, high)
        return shares * value

    def purchase(self, dollars):
        current_prices = self.input_data[self.current_index]
        high = current_prices[2]
        low =  current_prices[3]
        value_charged = random.uniform(low, high)
        shares = int((dollars - self.transaction_cost)/value_charged)
        if shares > 0:
            change = (dollars - self.transaction_cost) - (shares*value_charged)
            if change < 0.01:
                change = 0.0
            return shares, change
        else:
            return 0, dollars

    def sell_holdings(self, shares):
        current_prices = self.input_data[self.current_index]
        high = current_prices[2]
        low = current_prices[3]
        selling_price = random.uniform(low, high)
        return (selling_price * shares) - self.transaction_cost



if __name__ == '__main__':

    model = 'model_final4'
    TRANSACTION_COST = 10.0
    print('Creating market')
    market = Market(TRANSACTION_COST)
    print('Market data loaded')
    whole_start = pandas.datetime(2013, 11, 28)
    whole_end = pandas.datetime(2018, 11, 1)
    market.create_simulation_SandPonly(whole_start, whole_end)

    # load json and create model
    loaded_model_json = None
    with open('./{}.json'.format(model), 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights('./{}.h5'.format(model))
    print("Loaded model from disk")
    bob = Agent('bob', 100000.00, market, loaded_model)
    with open('portfolio_{}.txt'.format(model), 'w') as port_writer:
        in_progress = True
        while in_progress:
            info = market.get_current_data()
            if info is None:
                in_progress = False
            else:
                picks = bob.build_portfolio(info)[0]
                output = ''
                for component in picks:
                    output += str(component) + '\t'
                port_writer.write('{}\n'.format(output))


