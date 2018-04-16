import multiprocessing as mp
import h5py
import h5py_cache as h5c


class _Session:
    """
    Class abstracting the idea of sessions for hdf5. All hdf5 handling occurs here.
    """

    def __init__(self):
        self.lock_dict = {}
        self.lock = mp.Lock()

    def add_hdf5(self, filepath):
        """
        Adds an hdf5 to track if nonexistent, else gets it

        :param filepath: path to hdf5
        :return: hdf5 lock
        """
        with self.lock:
            if filepath not in self.lock_dict.keys():
                hdf5_lock = mp.Lock()
                self.lock_dict[filepath] = hdf5_lock
            return self.lock_dict[filepath]

    def write_hdf5(self, filepath, dataset_id, vid_data):
        """
        Writes data to an hdf5 file

        :param filepath: filepath of hdf5
        :param dataset_id: id of dataset in hdf5
        :param vid_data: data to write
        :return: None
        """
        hdf5_lock = self.add_hdf5(filepath)
        with hdf5_lock:
            h5py_file = h5c.File(filepath, 'a', libver='latest', chunk_cache_mem_size=(1024**2)*16)
            h5py_file.create_dataset(dataset_id, data=vid_data, dtype='uint8')
            h5py_file.close()

    def read_hdf5(self, filepath, dataset_id, start_frame, end_frame):
        """
        Reads data from an hdf5

        :param filepath: filepath of hdf5
        :param dataset_id: id of dataset in hdf5
        :param start_frame: frame to start from
        :param end_frame: frame to end before (exclusive)
        :return: data
        """
        hdf5_lock = self.add_hdf5(filepath)
        with hdf5_lock:
            h5py_file = h5c.File(filepath, 'r', libver='latest', chunk_cache_mem_size=(1024**2)*16)
            dataset = h5py_file[dataset_id]
            return dataset[start_frame: end_frame]


session = _Session()
