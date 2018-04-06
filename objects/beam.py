from .brick import Brick
from typing import Iterable
from abc import abstractmethod
from utils.config import Config
from ._session import session
try:
    import ujson as json
except:
    import json
import stringcase
import numpy as np


class Beam:
    def __init__(self, brick_class, json_dict: dict=None, datastring: str='', filepath: str='', config: Config=None,
                 demo: bool=False, fps: int=30):
        """
        A data structure that encapsulates a video of training data.
        Order of loading is <json_dict> => <datastring> => <filepath>.

        :param json_dict: data to initialize beam with
        :param datastring: data string to initialize beam with
        """
        super().__init__()
        json_dict = json_dict or {}
        self.brick_class = brick_class
        self.config = config or None
        self.filepath = filepath
        self.bricks = []
        self.valid_bricks = []
        self.name, self.id, self.filepath, self.source_path, = '', '', '', ''
        self.source_json_path, self.hdf5_path, self.dataset_id = '', '', ''
        self.fps = fps
        self.metadata = {}
        self._deserialize(json_dict=json_dict, datastring=datastring, filepath=filepath)

    def __len__(self):
        return len(self.valid_bricks)

    def __getitem__(self, item):
        return self.valid_bricks[item]

    def _serialize(self, filepath: str='', to_file=True):
        """
        Serializes the beam as a JSON and either returns it or saves to file.

        :param filepath: <str> path to serialize to
        :param to_file: <bool> whether to write to file
        :return: <None> or <str>
        """
        filepath = filepath or self.filepath
        assert not to_file or filepath, 'A <filepath> to serialize to must be provided in the constructor or as an argument'

        dict_repr = self.serialized()
        if to_file:
            with open(filepath, 'w+') as f:
                json.dump(dict_repr, f)
            return None
        else:
            return json.dumps(dict_repr)

    def _deserialize(self, json_dict: dict=None, datastring: str='', filepath: str=''):
        """
        Deserializer helper. Order of loading is <json_dict> => <datastring> => <filepath>.

        :param json_dict:
        :param datastring:
        :param filepath:
        :return:
        """
        assert json_dict or datastring or filepath, \
            'Please provide at least one of <json_dict>, <datastring>, or <filepath> to deserialize from.'
        if not json_dict:
            if datastring:
                json_dict = json.loads(datastring)
            else:
                with open(filepath, 'r') as f:
                    json_dict = json.load(f)

        self_dict = vars(self)
        for key, value in json_dict.items():
            if key == 'bricks':
                self_dict[key] = [self.brick_class(json_dict=value[i], beam=self, index=i) for i in range(len(value))]
            else:
                self_dict[stringcase.snakecase(key)] = value

    def serialize(self, filepath: str=''):
        return self._serialize(filepath=filepath, to_file=True)

    def serialized(self):
        dict_repr = {}
        for key, val in vars(self).items():
            if key == 'config' or key == 'brick_class':
                continue
            dict_repr[stringcase.camelcase(key)] = val
        dict_repr['bricks'] = [brick.serialized() for brick in dict_repr['bricks']]
        return dict_repr

    def serializes(self):
        return self._serialize(to_file=False)

    def deserialize(self, filepath: str):
        return self._deserialize(filepath=filepath)

    def deserializes(self, datastring: str):
        return self._deserialize(datastring=datastring)

    def deserialized(self, json_dict: dict):
        return self._deserialize(json_dict=json_dict)

    def add_brick(self, brick):
        """
        Adds a brick to the beam

        :param brick: brick to add
        :return: None
        """
        self.bricks.append(brick)
        brick.index = len(self.bricks) - 1
        brick.beam = self

    def get_bricks(self,
                   exclude_invalid=True,
                   sort_by=lambda b: b.index,
                   filter_by=lambda b: True,
                   filter_posthook=lambda bs: bs) -> Iterable[Brick]:
        # print(self.bricks)
        filter_by_internal = (lambda brick: brick.valid and filter_by(brick)) if exclude_invalid else filter_by
        bricks = sorted(filter(filter_by_internal, self.bricks), key=sort_by)
        self.valid_bricks = list(filter_posthook(list(bricks)))
        return list(self.valid_bricks)

    def write_data(self, vid_data):
        """
        Writes video data associated with this beam to a file.

        :param vid_data: <np.array> an array of data to write
        :return: None
        """
        if not self.hdf5_path:
            raise FileNotFoundError('HDF5 filepath is missing')
        if not self.dataset_id:
            raise FileNotFoundError('Dataset ID is missing')
        session.write_hdf5(self.hdf5_path, self.dataset_id, vid_data)

    def append_data(self):
        # TODO: Implement data appending
        pass

    def read_data(self, frame: int, fps: int, num_frames: int):
        """
        Reads a block of data associated with a video.

        :param frame: <int> frame associated with the first label
        :param fps: <int> fps to read the data at
        :param num_frames: <int> number of frames to include
        :return:
        """
        if not self.hdf5_path:
            raise FileNotFoundError('HDF5 filepath is missing')
        if not self.dataset_id:
            raise FileNotFoundError('Dataset ID is missing')
        stride = self.fps // fps
        read_frames = []
        for i in range(num_frames):
            num_frames_back = (i - num_frames + 1) * stride
            frame_ind = frame + num_frames_back
            read_frames.append(session.read_hdf5(self.hdf5_path, self.dataset_id, frame_ind, frame_ind+1))

        return np.concatenate(read_frames, axis=0)

    def reset_bricks(self, new_bricks=None):
        new_bricks = new_bricks or []
        self.bricks = list(new_bricks)

    def is_valid(self) -> bool:
        return self.valid

    def forward_brick_validity(self, frame: int, fps: int, num_frames: int) -> bool:
        stride = self.fps // fps
        return frame - ((num_frames - 1) * stride) >= 0

    def backward_brick_validity(self, frame: int, fps: int, num_frames: int) -> bool:
        stride = self.fps // fps
        return frame + ((num_frames - 1) * stride) < len(self.bricks)

    def get_forward_bricks(self, frame: int, fps: int, num_frames: int) -> bool:
        stride = self.fps // fps
        forward_bricks = []
        for i in range(1, num_frames):
            num_frames_forward = i * stride
            frame_ind = frame + num_frames_forward
            forward_bricks.append(self.bricks[frame_ind])
        return forward_bricks

    @property
    @abstractmethod
    def valid(self) -> bool:
        return True
