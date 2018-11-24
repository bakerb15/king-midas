
class Agent(object):

    def __init__(self, starting_dollars, market_simulation):
        self.dollars = starting_dollars
        self.market_simulation = market_simulation
        self.portfolio = {}
        self.knowledge = {}


