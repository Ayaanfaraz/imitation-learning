import pickle

filename = "_out/imitation_training_data_chunked.pkl"

with open("_out/imitation_training_data.pkl", 'rb') as af:
    str_to_save = pickle.load(af)

i = 0 
count  = 0
with open(filename,'wb') as file_handle:
    while i < len(str_to_save):
        pickle.dump(str_to_save[i:i+1], file_handle)
        i = i+1
        count = count + 1
        # print("i value is: ",i)
        # pickle.dump(str_to_save[4:10], file_handle)
        # pickle.dump(str_to_save[10:17], file_handle)

# file_handle = open(Path+filename,'rb')

# def getStates(fname):
#     return pickle.load(fname)

# print(count) 
# for i in range (count):
#     print(i)
#     result = getStates(file_handle)
#     print(len(result),"\n")
    

