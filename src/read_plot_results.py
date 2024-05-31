import pickle

def load_results(filename="store_simulation.pkl"):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data
def plot_results_simulation_real_world():
    www=1

if __name__ == '__main__':
    load_results(filename="store_simulation.pkl")
    load_results(filename="store_realworld.pkl")
    #load_results_realworld()