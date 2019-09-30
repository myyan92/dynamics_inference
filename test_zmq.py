import zmq

if __name__ == "__main__":
    context = zmq.Context()
    service = context.socket(zmq.DEALER)

    service.bind("tcp://127.0.0.1:8080")
    service.send_multipart([b"", b"hello"])
    service.send_multipart([b"", b"hello"])
    service.send_multipart([b"", b"hello"])

#    service.send(b"", zmq.SNDMORE)
#    service.send(b"hello world")
    reply = service.recv_multipart()
    print(reply)
    reply = service.recv_multipart()
    print(reply)
    reply = service.recv_multipart()
    print(reply)
    service.close()

#    stream = ZMQStream(service)
#    stream.on_recv(self.response_handler)
