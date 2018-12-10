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
        #prc = (today - yesterday)/ yesterday
        prc = today / yesterday
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

def create_dataset2(fulldataset, companyonlydataset,  selector, how_many_to_pick, ATTR_PER_COMPANY, forward=0, look_back=1, only_x=False):
  dataX, dataY = [], []

  # normalizing
  fulldataset_normalized = []
  last = 0
  for index in range(1, len(fulldataset)):
    last = index
    normalized = []
    observations =[[], [], [], [], [], []]
    totals = [[], [], [], [], [], []]
    current = fulldataset[index]
    previous = fulldataset[index - 1]
    averages = []

    for i in range(len(current)):
      try:
        yesterday = previous.item(i)
        today = current.item(i)
        prc = (today - yesterday)/ yesterday
        observations[i % 6].append(prc)
        totals[i % 6].append(prc)
      except ZeroDivisionError:
        observations[i % 6].append(0.0)
        totals[i % 6].append(0.0)

  for i in range(len(totals)):
    total = 0
    for num in totals[i]:
      total += num
    averages[i] = total/len(total[i])

  for index in range(len(observations)):
    observations[index] = sorted(observations[index])

  for i in range(len(normalized)):
    try:
      min_x = observations[i % 6][0]
      max_x = observations[i % 6][-1]
      numerator = min[x] - averages[i]
      denom = max_x - min_x
      x = (2 *(numerator/denom)) - 1
      normalized[i] = x
    except ZeroDivisionError:
      normalized[i] = 0



    fulldataset_normalized.append(np.array(normalized))

  fulldataset_normalized = pandas.DataFrame(fulldataset_normalized).values




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