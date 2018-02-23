import json
import stringcase
from abc import abstractmethod


class Brick:
    def __init__(self, json_dict: dict=None, datastring: str='', beam =None):
        """
        A data structure that encapsulates a trainable moment

        :param json_dict: <dict> data that brick holds
        :param datastring: <str> string serializing the json_dict
        :param beam: <Beam> beam that holds this brick
        """
        super().__init__()
        json_dict = json_dict or {}
        self.frame = 0
        self.truth, self.metadata, self.raw_metadata = {}, {}, {}
        self.deserialize(json_dict=json_dict, datastring=datastring)
        self.beam = beam

    def serialized(self) -> dict:
        """Get the serialized representation as a dictionary."""
        dict_repr = {}
        for key, val in vars(self).items():
            dict_repr[stringcase.camelcase(key)] = val
        del dict_repr['beam']
        return dict_repr

    def serialize(self) -> str:
        """Serializes the brick as a JSON string"""
        return json.dumps(self.serialized())

    def deserialize(self, json_dict=None, datastring=''):
        """
        Deserializes the brick

        :param json_dict: <dict> data that brick holds
        :param datastring: <str> string serializing the json_dict
        """
        assert bool(json_dict) ^ bool(datastring), 'Please provide exactly one of <json_dict> or <datastring>'
        json_dict = json_dict or json.loads(datastring)
        self_dict = vars(self)
        for key, value in json_dict.items():
            self_dict[stringcase.snakecase(key)] = value

    def is_valid(self) -> bool:
        return self.valid

    @property
    @abstractmethod
    def valid(self) -> bool:
        pass

    @abstractmethod
    def get_input(self):
        pass

    @abstractmethod
    def get_metadata(self):
        pass

    @abstractmethod
    def get_truth(self):
        pass
