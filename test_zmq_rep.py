import zmq

if __name__ == "__main__":
    context = zmq.Context()
    service = context.socket(zmq.REP)

    service.connect("tcp://instance-1:8080")
    request = service.recv()
    print(request)
    service.send(b"start working 1!")
    request = service.recv()
    print(request)
    service.send(b"start working 2!")
    request = service.recv()
    print(request)
    service.send(b"start working 3!")

    service.close()
