
import pickle

# Specify the path to your pickle file
pickle_file = 'init_plan_trial.pickle'

# Open the file in 'rb' mode (read binary)
with open(pickle_file, 'rb') as file:
    data = pickle.load(file)

# Now 'data' contains the deserialized object
print(data)
