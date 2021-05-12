FROM golang:1.14

RUN mkdir consumer
COPY consumer/ /home/consumer/

COPY go.mod /home/
COPY go.sum /home/

WORKDIR /home/

RUN go mod download

RUN CGO_ENABLED=0
