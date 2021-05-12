package main

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"fmt"
	"html/template"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	pb "hotaru/protos"

	"github.com/labstack/echo"
	"github.com/streadway/amqp"
	"google.golang.org/grpc"
)

const (
	address     = "server:50051"
	amqpaddress = "amqp://rabbitmq:5672"
	defaultName = "world"
	title       = "Hotaru"
)

var conn *grpc.ClientConn
var client pb.HelloClient
var amqpCh *amqp.Channel

type Template struct {
	templates *template.Template
}

type (
	illust struct {
		Data string `json:"data"`
	}
)

func (t *Template) Render(w io.Writer, name string, data interface{}, c echo.Context) error {
	return t.templates.ExecuteTemplate(w, name, data)
}

func failOnError(err error, msg string) {
	if err != nil {
		log.Fatalf("%s: %s", msg, err)
		panic(fmt.Sprintf("%s: %s", msg, err))
	}
}

func setup() {
	Conn, err := grpc.Dial(address, grpc.WithInsecure(), grpc.WithBlock())
	conn = Conn
	failOnError(err, "Failed to connect gRPC server")
	C := pb.NewHelloClient(conn)
	client = C

	fmt.Println("connected!")
}

func setupAMQP() {
	conn, err := amqp.Dial(amqpaddress)
	failOnError(err, "Failed to connect to MQ")

	channel, err := conn.Channel()
	failOnError(err, "Failed to open a channel")

	amqpCh = channel
}

func main() {
	t := &Template{
		templates: template.Must(template.ParseGlob("templates/*.html")),
	}

	setup()
	setupAMQP()

	data := struct {
		Title string
		Img   string
	}{
		Title: title,
		Img:   "echo",
	}

	e := echo.New()
	e.Static("/static", "static")
	e.Renderer = t
	e.GET("/", func(c echo.Context) error {
		return c.Render(http.StatusOK, "start", data)
	})
	e.POST("/run", manupilate)
	e.Logger.Fatal(e.Start(":1323"))
}

func manupilate(c echo.Context) error {
	jsonMap := make(map[string]string)
	err := json.NewDecoder(c.Request().Body).Decode(&jsonMap)
	failOnError(err, "Failed to extract from body json")
	src_str := jsonMap["src_uri"]
	ref_str := jsonMap["ref_uri"]

	src_str = extractBase64(src_str)
	ref_str = extractBase64(ref_str)

	ctx, _ := context.WithTimeout(context.Background(), 5*time.Second)

	resp, err := client.ImageManupilate(ctx, &pb.ImgMessage{Src: src_str, Ref: ref_str})
	failOnError(err, "Failed to get results from gRPC server")

	data := illust{Data: resp.Img}

	filename, _ := createRandomStr(12)

	addMessage("grpc-queue", resp.Img, filename+"_predict")
	addMessage("grpc-queue", src_str, filename+"_src")
	addMessage("grpc-queue", ref_str, filename+"_ref")

	return c.JSON(http.StatusOK, data)
}

func addMessage(queueName string, entity string, filename string) {
	data := "{entity:" + entity + ",name:" + filename + "}"
	json_data := []byte(data)
	err := amqpCh.Publish(
		"",
		queueName,
		false,
		false,
		amqp.Publishing{
			ContentType: "application/json",
			Body:        json_data,
		},
	)
	failOnError(err, "Failed to add message to queue")
}

func extractBase64(str string) string {
	str = strings.Replace(str, "data:image/png;base64,", "", -1)
	str = strings.Replace(str, " ", "+", -1)

	return str
}

func createRandomStr(digit uint32) (string, error) {
	const letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

	b := make([]byte, digit)
	if _, err := rand.Read(b); err != nil {
		fmt.Println(err)
	}

	var rnd string

	for _, v := range b {
		rnd += string(letters[int(v)%len(letters)])
	}

	return rnd, nil
}
