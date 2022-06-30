import os
import pickle

combined_data = []
# Iterate over the 6 data files
for i in range(1,7):

    individual_data = []
    filename = '_out/images_' + str(i) + '.pkl'
    print("Processing filename: " + filename)

    # Store in data from each file into a sub list
    with open(filename, 'rb') as f:
        individual_data = pickle.load(f)

    for data in individual_data:
        combined_data.append(data)

# Save final concatenated data files        
with open("_out/imitation_training_images.pkl", "wb") as of:
    pickle.dump(combined_data,of)

print("Number of training data points: " + str(len(combined_data)))
print("Finished with size: " + str(os.path.getsize("_out/imitation_training_images.pkl")))
