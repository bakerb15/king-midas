import numpy as np

def create_dataset(fulldataset, companyonlydataset,  selector, how_many_to_pick, ATTR_PER_COMPANY, forward=0, look_back=1):
  dataX, dataY = [], []
  for i in range(len(fulldataset)-look_back-forward+1):
    a = fulldataset[i:(i+look_back), :]

    c = companyonlydataset[:(i+look_back), :]
    d = companyonlydataset[i + forward+look_back - 1, :]

    b = selector(c, d, how_many_to_pick, ATTR_PER_COMPANY)


    dataX.append(a)
    dataY.append(b)
  return np.array(dataX), np.array(dataY)