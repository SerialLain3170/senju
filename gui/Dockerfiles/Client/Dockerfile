FROM golang:1.14

RUN mkdir client
COPY client/ /home/client/

RUN mkdir protos
COPY protos/ /home/protos/

RUN mkdir templates
COPY templates/ /home/templates/

RUN mkdir static
COPY static/ /home/static/

COPY go.mod /home/
COPY go.sum /home/
COPY client/start_client.sh /home/

WORKDIR /home/

RUN go mod download

RUN CGO_ENABLED=0
