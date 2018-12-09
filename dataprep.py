import numpy as np
import pandas

def create_dataset(fulldataset, companyonlydataset,  selector, how_many_to_pick, ATTR_PER_COMPANY, forward=0, look_back=1, only_x=False):
  dataX, dataY = [], []

  # normalizing
  fulldataset_normalized = []
  last = 0
  for index in range(1, len(fulldataset)):
    last = index
    normalized = []
    current = fulldataset[index]
    previous = fulldataset[index - 1]
    for i in range(len(current)):
      try:
        yesterday = previous.item(i)
        today = current.item(i)
        prc = (today - yesterday)/ yesterday
        if prc > 1.0:
          prc = 1.0
        elif prc < -1.0:
          prc = -1.0
        normalized.append(prc)
      except ZeroDivisionError:
        normalized.append(0.0)
    fulldataset_normalized.append(np.array(normalized))

  fulldataset_normalized = pandas.DataFrame(fulldataset_normalized).values


  #   shift the company only dataset forward by one because we can use first date
  # because now using percent change between two frames

  if companyonlydataset is not None:
    companyonlydataset = companyonlydataset[1:]

  for i in range(len(fulldataset_normalized) - (look_back + forward)+1):
    a = fulldataset_normalized[i:(i+look_back), :]

    if companyonlydataset is not None:
      c = companyonlydataset[:(i+look_back), :]
      d = companyonlydataset[i + forward+look_back - 1, :]

      b = selector(c, d, how_many_to_pick, ATTR_PER_COMPANY)
      dataY.append(b)

    dataX.append(a)

  if only_x:
    return np.array(dataX)
  else:
    return np.array(dataX), np.array(dataY)