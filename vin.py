import pickle
with open("_out/imitation_training_images.pkl","rb") as f:
    result = pickle.load(f)
    print("type: ",type(result))
    print("length: ",len(result))
    print("shape: ",result.shape)
