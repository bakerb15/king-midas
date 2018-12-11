from market import Market
from agent import Agent, PassiveAgent
import pandas
from keras.models import model_from_json

#the name of the model used for the active agent
#make sure .h5 and .json are named the same
model = 'model_final4'


TRANSACTION_COST = 10.0
AGENTS_TO_SIMULATE = 20
STARTING_DOLLARS = 100000.00
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

active_agents = []
for i in range(AGENTS_TO_SIMULATE):
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights('./{}.h5'.format(model))
    active_agents.append(Agent('active_{}'.format(str(i)), STARTING_DOLLARS, market, loaded_model))
passive_agent = PassiveAgent('passive', STARTING_DOLLARS,  market, loaded_model)
with open('simulation_results_20agents_{}.csv'.format(model), 'w') as port_writer:
    port_writer.write('Passive_Agent_Net_Worth,\t Passive_Agent_Shares,\t')
    for agent in active_agents:
        port_writer.write('{},\t {},\t '.format(agent.name + 'Net_Worth', agent.name + 'Shares'))
    port_writer.write('\n')
    in_progress = True
    while in_progress:
        in_progress = market.step()
        port_writer.write('{},\t {},\t '.format(passive_agent.net_worth(), passive_agent.shares))
        for agent in active_agents:
            port_writer.write('{},\t {},\t '.format(agent.net_worth(), agent.shares))
        port_writer.write('\n')

