from market_data import company, market_index
import os
from enum import Enum
import pandas
import datetime
from collections import OrderedDict
from agent import Agent

class Market(object):

    def __init__(self, transaction_cost):
        self.companies = OrderedDict()
        self.market_indices = OrderedDict()
        self.transaction_cost = transaction_cost
        for key in company:
            self.companies[company[key]] = {'name' : key}
        for key in market_index:
            self.market_indices[market_index[key]] = {'name' : key}
        sorted(self.companies)
        sorted(self.market_indices)
        data_dir = './data'
        for file_name in os.listdir(data_dir):
            if '.csv' in file_name:
                path_to_file = os.path.join(data_dir, file_name)
                symbol = file_name.split('.csv')[0]
                if symbol in self.companies:
                    print('Loading {}'.format(self.companies[symbol]['name']))
                    self.companies[symbol]['data'] = pandas.read_csv(path_to_file, index_col=0)

                elif symbol in self.market_indices:
                    print('Loading {}'.format(self.market_indices[symbol]['name']))
                    self.market_indices[symbol]['data'] = pandas.read_csv(path_to_file, index_col=0)
                else:
                    print('{} is not in market_data dictionaries.'.format(symbol))

    def simulate(self, start_date, end_date, agents):
        print('Starting simulation for time period {} through {}.'.format(start_date, end_date))
        length = end_date - start_date
        if length.days > 0:
            day = start_date
            while day < end_date:
                info = {}
                date = day.strftime('%Y-%m-%d')
                for ind in self.market_indices:
                    if date in self.market_indices[ind]['data'].index:
                        info[ind] = self.market_indices[ind]['data'].loc[date]
                for co in self.companies:
                    try:
                        if date in self.companies[co]['data'].index:
                                info[co] = self.companies[co]['data'].loc[date]
                    except KeyError:
                        break;
                for ag in agents:
                    ag.observe(day, info)
                for ag in agents:
                    print('{}\t{}\t{}'.format(date, ag.name, ag.calculate_net_worth()))
                day += datetime.timedelta(days=1)

if __name__ == '__main__':
    TRANSACTION_COST = 10.0
    print('Creating market')
    mark = Market(TRANSACTION_COST)
    print('Market data loaded')
    whole_start = datetime.date(2009, 11, 3)
    whole_end = datetime.date(2012, 12, 12)
    agents = [Agent('sally', 100000.0, mark)]
    mark.simulate(whole_start, whole_end, agents)

