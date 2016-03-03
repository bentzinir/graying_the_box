import h5py

def load_hdf5(object_name, path, num_frames=None):
    print "    Loading " +  object_name
    obj_file = h5py.File(path + object_name + '.h5', 'r')
    obj_mat = obj_file['data']
    return obj_mat[:num_frames]

def save_hdf5(object_name, path, object):
    print "    Saving " +  object_name
    with h5py.File(path + object_name +'.h5', 'w') as hf:
        hf.create_dataset('data', data=object)