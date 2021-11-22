FROM golang:1.14

RUN mkdir {client,protos,templates,static}

COPY client/ /home/client/
COPY protos/ /home/protos/
COPY templates/ /home/templates/
COPY static/ /home/static/
COPY go.mod /home/
COPY go.sum /home/
COPY client/start_client.sh /home/

WORKDIR /home/

RUN go mod download

RUN CGO_ENABLED=0
