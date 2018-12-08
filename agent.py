from collections import deque
from keras.models import model_from_json

class Agent(object):

    def __init__(self, agent_name, starting_dollars, market_simulation):
        self.name = agent_name
        self.dollars = starting_dollars
        self.market_simulation = market_simulation
        self.portfolio = {}
        self.knowledge = deque()
        self.MAX_KNOWLEDGE = 500
        self.last_company_price = {}
        self.last_market_index_price = {}

    def observe(self, date, info):
        if len(self.knowledge) < self.MAX_KNOWLEDGE:
            self.knowledge.append((date, info))
        else:
            self.knowledge.popleft()
            self.knowledge.append((date, info))
        for key in info:
            if key in self.market_simulation.companies:
                self.last_company_price[key] = info[key]['Close']
            elif key in self.market_simulation.market_indices:
                self.last_market_index_price[key] = info[key]['Close']

    def act(self):
        pass

    def calculate_net_worth(self):
        nw = self.dollars
        for stock in self.portfolio:
            nw += self.last_company_price[stock]
        return nw






