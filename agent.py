from collections import deque
import pandas
import random

#last day in data 2018-11-01
CURRENT_DATE = pandas.datetime(2016, 11, 28) # last day in training set

MAX_KNOWLEDGE = 10

class Agent(object):

    def __init__(self, agent_name, starting_dollars, market_simulation, model):
        self.name = agent_name
        self.dollars = starting_dollars
        self.shares = 0
        self.market_simulation = market_simulation
        self.market_simulation.add_me(self)
        self.knowledge = deque()

        self.last_company_price = {}
        self.last_market_index_price = {}
        self.model = model
        self.avg_signal = 0
        self.steps = 0

    def observe(self, info):
        self.steps += 1
        current_signal = 0
        if len(self.knowledge) > 0:
            discount = 1/len(self.knowledge)
        else:
            discount = 0
        count = 1
        for item in self.knowledge:
            current_signal += (count * discount * item)
            count += 1
        if len(self.knowledge) > 0:
            current_signal = current_signal/len(self.knowledge)
        if self.steps >= MAX_KNOWLEDGE:
            self.act(current_signal)
        if len(self.knowledge) < MAX_KNOWLEDGE:
            self.knowledge.append(self.model.predict(info, batch_size=1)[0])
        else:
            self.knowledge.popleft()
            self.knowledge.append(self.model.predict(info, batch_size=1)[0])




    def net_worth(self):
        return self.dollars + self.market_simulation.value_of_shares(self.shares)

    def act(self, new_signal):
        #if new average signal is more than avg signal buy random amount
        if new_signal > self.avg_signal:
            self.buy()
        else: #if new average signal is more than avg signal sell random amount
            self.sell()

    def buy(self):
        percent_to_spend =  random.uniform(0.0, 1.0)
        dollars_to_market = (percent_to_spend * self.dollars)
        self.dollars -= dollars_to_market
        new_shares, change  = self.market_simulation.purchase(dollars_to_market)
        self.shares += new_shares
        self.dollars += change

    def sell(self):
        percent_to_sell = random.uniform(0.0, 1.0)
        shares_to_sell = int(percent_to_sell * self.shares)
        if shares_to_sell > 0:
            self.dollars += self.market_simulation.sell_holdings(int(percent_to_sell*self.shares))
            self.shares -= int(percent_to_sell*self.shares)


    def build_portfolio(self, nn_input):
        return self.model.predict(nn_input, batch_size=1)


class PassiveAgent(object):

    def __init__(self, agent_name, starting_dollars, market_simulation, model):
        self.name = agent_name
        self.dollars = starting_dollars
        self.shares = 0
        self.market_simulation = market_simulation
        self.market_simulation.add_me(self)
        self.knowledge = deque()
        self.MAX_KNOWLEDGE = 30
        self.last_company_price = {}
        self.last_market_index_price = {}
        self.model = model
        self.avg_signal = 0
        self.has_purchased = False
        self.steps = 0

    def observe(self, info):
        self.steps += 1
        if self.steps >= MAX_KNOWLEDGE:
            if self.has_purchased is False:
                self.buy()
                self.has_purchased = True



    def net_worth(self):
        return self.dollars + self.market_simulation.value_of_shares(self.shares)

    def act(self, new_signal):
        # if new average signal is more than avg signal buy random amount
        if new_signal > self.avg_signal:
            self.buy()
        else:  # if new average signal is more than avg signal sell random amount
            self.sell()

    def buy(self):
        percent_to_spend = 1.0
        dollars_to_market = (percent_to_spend * self.dollars)
        self.dollars -= dollars_to_market

        new_shares, change = self.market_simulation.purchase(dollars_to_market)
        self.shares += new_shares
        self.dollars += change

    def sell(self):
        percent_to_sell = random.random(0.0, 1.0)
        self.dollars += self.market_simulation.sell_holdings(int(percent_to_sell * self.shares))
        self.shares -= int(percent_to_sell * self.shares)

    def build_portfolio(self, nn_input):
        return self.model.predict(nn_input, batch_size=1)




