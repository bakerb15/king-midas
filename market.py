from market_data import company, market_index
import os
from enum import Enum
import pandas
import datetime

class Market(object):

    def __init__(self):
        self.companies = {}
        self.market_indices = {}
        for key in company:
            self.companies[company[key]] = {'name' : key}
        for key in market_index:
            self.market_indices[market_index[key]] = {'name' : key}
        data_dir = './data'
        for file_name in os.listdir(data_dir):
            if '.csv' in file_name:
                path_to_file = os.path.join(data_dir, file_name)
                symbol = file_name.split('.csv')[0]
                if symbol in self.companies:
                    print('Loading {}'.format(self.companies[symbol]['name']))
                    self.companies[symbol]['data'] = pandas.read_csv(path_to_file)
                elif symbol in self.market_indices:
                    print('Loading {}'.format(self.market_indices[symbol]['name']))
                    self.market_indices[symbol]['data'] = pandas.read_csv(path_to_file)
                else:
                    print('{} is not in market_data dictionaries.'.format(symbol))

    def simulate(self, start_date, end_date):
        print('Starting simulation for time period {} through {}.'.format(start_date, end_date))
        length = end_date - start_date
        print('Length of simulation {}.'.format(length.days))

if __name__ == '__main__':
    print('Creating market')
    market = Market()
    print('Market data loaded')
    whole_start = datetime.date(2008, 11, 1)
    whole_end = datetime.date(2018, 11, 1)
    market.simulate(whole_start, whole_end)
