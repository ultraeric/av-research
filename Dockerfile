# Base Image
FROM python:3.5.4-onbuild

# Command to run, if any
RUN pip3 install http://download.pytorch.org/whl/cu90/torch-0.3.0.post4-cp35-cp35m-linux_x86_64.whl
RUN pip3 install torchvision