FROM python:3.9

RUN apt-get update && apt-get install -y build-essential python-dev libgl1-mesa-dev nano telnet
RUN pip3 install opencv-python Pillow numpy pyyaml grpcio grpcio-tools
RUN pip3 install --no-cache-dir torch torchvision

RUN mkdir {server,protos}

COPY server/ /home/server/
COPY server/ /home/protos/

WORKDIR /home/server/
