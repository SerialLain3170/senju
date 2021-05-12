docker build -t goclient -f ./Dockerfiles/Client/Dockerfile .
docker build -t pyserver -f ./Dockerfiles/Server/Dockerfile .
docker build -t goconsumer -f ./Dockerfiles/Consumer/Dockerfile .
docker build -t rabbitmq -f ./Dockerfiles/RabbitMQ/Dockerfile .
docker build -t postgres -f ./Dockerfiles/PostgreSQL/Dockerfile .
docker build -t nginx -f ./Dockerfiles/Nginx/Dockerfile .