from market import Market
from agent import Agent, PassiveAgent
import pandas
from keras.models import model_from_json

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
active_agent = Agent('bob', 100000.00, market, loaded_model)
passive_agent = PassiveAgent('passive', 100000.00, market, loaded_model)
with open('simulation_results_{}_act_before_signal.txt'.format(model), 'w') as port_writer:
    port_writer.write('Passive_Agent_Net_Worth, Passive_Agent_Shares,  Active_Agent_Net_Worth, Active_Agent_Shares\n')
    in_progress = True
    while in_progress:
        in_progress = market.step()
        port_writer.write('{},\t {},\t {},\t {}\n'.format(passive_agent.net_worth(), passive_agent.shares, active_agent.net_worth(), active_agent.shares))

