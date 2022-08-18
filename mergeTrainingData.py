import os
import pickle

combined_data = []
combined_image = []

for i in [47, 55]:

    individual_data = []
    individual_image = []
    dataname = '_out/imitation_training_data_' + str(i) + 'k.pkl'
    imagename = '_out/imitation_training_images_' + str(i) + 'k.pkl'
    print("Processing filename: " + dataname)
    print("Processing imagename: " + imagename)

    # Store in data from each file into a sub list
    with open(dataname, 'rb') as of:
        individual_data = pickle.load(of)
    with open(imagename, 'rb') as f:
        individual_image = pickle.load(f)

    for data in individual_data:
        combined_data.append(data)
    for image in individual_image:
        combined_image.append(image)

# Save final concatenated data files        
with open("_out/imitation_training_data_100k.pkl", "wb") as of:
    pickle.dump(combined_data,of)
with open("_out/imitation_training_images_100k.pkl", "wb") as f:
    pickle.dump(combined_image,f)

print("Number of training data points: " + str(len(combined_data)))
print("Finished with size: " + str(os.path.getsize("_out/imitation_training_data_100k.pkl")))
