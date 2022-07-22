#import os
#import pickle

#combined_data = []
## Iterate over the 6 data files
#for i in range(1,9):

    #individual_data = []
    #filename = '_out/data_' + str(i) + '.pkl'
    #print("Processing filename: " + filename)

    ## Store in data from each file into a sub list
    #with open(filename, 'rb') as f:
        #individual_data = pickle.load(f)

    #for data in individual_data:
        #combined_data.append(data)

#print(combined_data[0])
## Save final concatenated data files
#with open("_out/imitation_training_data.pkl", "wb") as of:
    #pickle.dump(combined_data,of)

#print("Number of training data points: " + str(len(combined_data)))
#print("Finished with size: " + str(os.path.getsize("_out/imitation_training_data.pkl")))

#with open("_out/imitation_training_data.pkl", 'rb') as f:
    #individual_data = pickle.load(f)

#with open("_out/test.pkl", "wb") as lf:
    #for data in individual_data:
        #pickle.dump(data, lf);

#with open("_out/test.pkl", 'rb') as af:
    #print(pickle.load(af)[1])

import os
import pickle

combined_data = 0
# Iterate over the 6 data files

individual_data = []
filename = '_out/imitation_training_images.pkl' #alis filenames
print("Processing filename: " + filename)

# Store in data from each file into a sub list
with open(filename, 'rb') as f:
    individual_data = pickle.load(f)

with open("_out/imitation_training_images_old_optimized.pkl", "wb") as of:
    for data in individual_data:
        #if combined_data < 3:
            #print( "data: " + str (data))
        combined_data += 1
        pickle.dump(data, of)

print("Number of training data points: " + str(combined_data))
print("Finished with size: " + str(os.path.getsize("_out/imitation_training_images_old_optimized.pkl")))

#print("Getting first element of merged data: ")
#with open("_out/imitation_training_data_optimized.pkl", 'rb') as f:
    #while True:
        #try:
            #print(pickle.load(f)[0])
        #except EOFError:
            #break
