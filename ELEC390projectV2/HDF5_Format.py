import h5py

def print_structure(name, obj):
    print(name)

with h5py.File('data.h5', 'r') as f:
    f.visititems(print_structure)
