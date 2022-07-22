import matplotlib.pyplot
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pickle
import collections

actions_list = []
with open('_out/imitation_training_data_optimized.pkl','rb') as af:
    while True:
        try:
            actions_list.append(pickle.load(af))
        except:
            break;


steer_list = [round(data[0],2) for data in actions_list]

dict = {}

for i in range(len(steer_list)):
    if steer_list[i] in dict:
        dict[steer_list[i]] += 1
    else:
        dict[steer_list[i]] = 1


labels = list(dict.keys())
freq = list(dict.values())

import matplotlib.pyplot as plt
plt.scatter(labels, freq)
plt.show()
