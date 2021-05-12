FROM python:3.9

RUN apt-get update && apt-get install -y build-essential python-dev libgl1-mesa-dev
RUN apt-get install -y nano
RUN apt-get install -y telnet

RUN pip3 install opencv-python Pillow
RUN pip3 install numpy pyyaml
RUN pip3 install --no-cache-dir torch torchvision
RUN pip3 install grpcio grpcio-tools

RUN mkdir server
COPY server/ /home/server/

RUN mkdir protos
COPY server/ /home/protos/

WORKDIR /home/server/
