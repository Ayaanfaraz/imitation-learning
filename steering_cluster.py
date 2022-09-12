import matplotlib.pyplot
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pickle
import collections

actions_list = []
with open('/home/asf170004/data/1million_data.pkl','rb') as af:
    for i in range(759403):
        actions_list.append(pickle.load(af))

print("Length of Data: ", len(actions_list))
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
plt.savefig("/home/asf170004/imitation_learning/_out/1millionSteering.png")
#plt.show()
