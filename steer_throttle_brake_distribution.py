import matplotlib.pyplot
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pickle
import collections

with open('_out/imitation_training_data.pkl','rb') as af:
    actions_list = pickle.load(af)

steer_list = [round(data[0],2) for data in actions_list]
throttle_list = [data[1] for data in actions_list]
brake_list = [data[2] for data in actions_list]
#print(steer_list)

dict = {"steer:-1":0, "steer:0":0, "steer:1":0,
        "brake:0":0, "brake:1":0,
        "throttle:0":0, "throttle:1":0}

# 0.0, 0.01, 0.015, 0.2, 0.35, 0.5, 0.7, 1
for i in range(len(steer_list)):

    if steer_list[i] >= -1 and steer_list[i] < -0.3:
        dict["steer:-1"] = dict["steer:-1"] + 1
    if steer_list[i] >= -0.3 and steer_list[i] <= 0.3:
        dict["steer:0"] = dict["steer:0"] + 1
    if steer_list[i] > 0.3 and steer_list[i] <= 1:
        dict["steer:1"] = dict["steer:1"] + 1

    if throttle_list[i] >0.3 and throttle_list[i] <= 1:
        dict["throttle:1"] = dict["throttle:1"] + 1
    if throttle_list[i] >=0 and throttle_list[i] <= 0.3:
        dict["throttle:0"] = dict["throttle:0"] + 1

    if brake_list[i] >0.3 and brake_list[i] <= 1:
        dict["brake:1"] = dict["brake:1"] + 1
    if brake_list[i] >=0 and brake_list[i] <= 0.3:
        dict["brake:0"] = dict["brake:0"] + 1

labels = list(dict.keys())
freq = list(dict.values())
#print(len(freq))
#for i in range(len(freq)):
#    freq[i] = float((freq[i]/len(actions_list))*100)

print(labels)
print(freq)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(labels,freq)
ax.set_ylabel("Distribution")
ax.set_title("Imitation Learning Training Data Distribution")
plt.show()
