import torch.nn as nn
import torch
import math

class NTM(nn.Module):
    """Implementation of NTM module"""

    def __init__(self, controller, reader, writer, memory, output, n_frames=2, n_steps=10):
        """

        :param controller: controller module that is called at each timestep. Must take [past, current] arguments.
        :param reader: takes in a vector and outputs a read vector.
        :param writer: takes in a vector and outputs an index vector and a content vector.
        :param memory: is read from and written to
        :param output: transforms read output into
        :param n_frames: number of input frames
        :param n_steps: number of output steps
        """
        super(NTM, self).__init__()
        self.controller, self.reader, self.writer, self.memory, self.output, self.n_frames, self.n_steps = \
            controller, reader, writer, memory, output, n_frames, n_steps

    def forward(self, camera, metadata):
        """

        :param camera: [batch x n_frames x height x width x depth]
        :param metadata: [batch x height x width x depth]
        :return: output controls [batch x n_steps x 2]
        """
        frame_steps = torch.unbind(camera, dim=1)
        previous_timestep = None
        for frame in frame_steps:
            controller_output, previous_timestep = self.controller(frame, previous_timestep)
