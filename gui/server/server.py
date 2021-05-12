import cv2 as cv
import numpy as np
import base64
import io
import time
import grpc
import hello_pb2
import hello_pb2_grpc

from concurrent import futures
from PIL import Image
from inference import Inferer


class HelloServer(hello_pb2_grpc.HelloServicer):
    def __init__(self):
        self.infer = Inferer()

        print("Model load finish!")

    @staticmethod
    def _decode(img_str: str) -> np.array:
        img_binary = base64.b64decode(img_str)
        img = np.frombuffer(img_binary, dtype=np.uint8)
        img = cv.imdecode(img, cv.IMREAD_COLOR)

        return img

    @staticmethod
    def _encode(img: np.array) -> str:
        pillow_object = Image.fromarray(img.astype(np.uint8))
        byte = io.BytesIO()
        pillow_object.save(byte, 'PNG')

        byte = byte.getvalue()
        base64_encoded = base64.b64encode(byte)

        return base64_encoded

    def ImageManupilate(self, request, context):
        print("Request accept!")

        src = self._decode(request.src)
        ref = self._decode(request.ref)
        img = self.infer(src, ref)
        base64_encoded = self._encode(img)

        print("Prediction finish!!")

        return hello_pb2.ImgResponse(img=base64_encoded)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor())
    hello_pb2_grpc.add_HelloServicer_to_server(HelloServer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print('Starting gRPC sample server...')

    try:
        while True:
            time.sleep(3600)

    except KeyboardInterrupt:
        server.stop(0)

if __name__ == "__main__":
    serve()