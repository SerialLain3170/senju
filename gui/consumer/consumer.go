package main

import (
	"database/sql"
	"encoding/base64"
	"fmt"
	"log"
	"os"
	"strings"

	_ "github.com/lib/pq"
	"github.com/streadway/amqp"
)

var db *sql.DB

const (
	amqpaddress = "amqp://rabbitmq:5672"
)

func failOnError(err error, msg string) {
	if err != nil {
		log.Fatalf("%s: %s", msg, err)
		panic(fmt.Sprintf("%s: %s", msg, err))
	}
}

func dbSetup() {
	Db, err := sql.Open("postgres", "host=postgres port=5432 user=Iwakura password=lain dbname=hotaru sslmode=disable")
	db = Db
	failOnError(err, "Failed to open DB")

	query := "create table if not exists sheet (fileid varchar(20))"
	_, err = db.Exec(query)
	failOnError(err, "Failed to create table")

	if _, err := os.Stat("img"); os.IsNotExist(err) {
		os.Mkdir("static", 0777)
	}
}

func main() {
	dbSetup()

	conn, err := amqp.Dial(amqpaddress)
	failOnError(err, "Failed to connect to MQ in consumer")

	channel, err := conn.Channel()
	failOnError(err, "Failed to open a channel in consumer")

	q, err := channel.QueueDeclare(
		"grpc-queue",
		false,
		false,
		false,
		false,
		nil,
	)
	failOnError(err, "Failed to declare a queue")

	messages, err := channel.Consume(
		q.Name,
		"",
		true,
		false,
		false,
		false,
		nil,
	)
	failOnError(err, "Failed to register a consumer")

	forever := make(chan bool)

	go func() {
		for data := range messages {
			image, name := stringSplit(string(data.Body))
			base64ToFilename(image, name)
			log.Printf("Save image succeeded!")
		}
	}()

	log.Printf("[INFO] Waiting for messages.....")
	<-forever
}

func stringSplit(str string) (string, string) {
	slice := strings.Split(str, ",")
	images := strings.Split(slice[0], ":")
	names := strings.Split(slice[1], ":")

	return images[1], names[1][:len(names[1])-1]
}

func base64ToFilename(image string, name string) {
	data, _ := base64.StdEncoding.DecodeString(image)

	file, _ := os.Create("img/" + name + ".png")
	defer file.Close()

	file.Write(data)

	if name[len(name)-5:] == "front" {
		query := "INSERT INTO sheet (fileid) VALUES ($1)"
		_, err := db.Exec(query, name[:len(name)-6])
		failOnError(err, "Failed to insert into DB")
	}
}
