import argparse
import os
import signal
import subprocess
import time

import requests


class Server:
    def __init__(self) -> None:
        raise NotImplementedError

    def wait(self):
        raise NotImplementedError

    def close(self):
        ret_code = self.process.poll()
        if ret_code is None:
            print(f"Killing the process for {type(self).__name__}")
            os.kill(self.process.pid, signal.SIGTERM)
        print(f"Closing the {type(self).__name__} instance")
        if self.output_file is not None:
            self.output_file.close()
        time.sleep(5)


class VLLMServer(Server):
    def __init__(self, model_path="") -> None:
        print(f"Starting the VLLM server with {model_path=}")
        self.is_ready = False
        self.model_path = model_path
        self.output_file = open("vllm_server.log", "w")

        self.process = subprocess.Popen(
            [
                "vllm",
                "serve",
                model_path,
            ],
            stdout=self.output_file,
            stderr=self.output_file,
        )

    def wait(self):
        if self.is_ready:
            return
        while True:
            time.sleep(5)
            try:
                response = requests.get("http://0.0.0.0:8000/health")
                if response.status_code == 200:
                    print("VLLMServer is ready")
                    self.is_ready = True
                    return
            except requests.exceptions.ConnectionError:
                print("VLLMServer is not ready yet, waiting for 5 seconds")


class ProxyServer(Server):
    def __init__(self, model_path="", approach="", return_full_response=True) -> None:
        print(f"Starting the ProxyServer with {approach=}")
        self.is_ready = False
        self.model_path = model_path
        self.approach = approach
        self.return_full_response = return_full_response
        self.output_file = open("proxy_server.log", "w")

        self.process = subprocess.Popen(
            [
                "python",
                "optillm.py",
                "--base_url=http://localhost:8000/v1",
                f"--model={self.model_path}",
                f"--approach={self.approach}",
                f"--return-full-response={self.return_full_response}",
            ],
            stdout=self.output_file,
            stderr=self.output_file,
        )

    def wait(self):
        if self.is_ready:
            return
        while True:
            time.sleep(5)
            try:
                response = requests.get("http://0.0.0.0:8080/health")
                if response.status_code == 200:
                    print("ProxyServer is ready")
                    self.is_ready = True
                    return
            except requests.exceptions.ConnectionError:
                print("ProxyServer is not ready yet, waiting for 5 seconds")


class ServerContext:
    def __init__(self, server_class, kwargs) -> None:
        self.server_class = server_class
        self.kwargs = kwargs

    def __enter__(self):
        print("Entering the context")
        self.server = self.server_class(**self.kwargs)
        return self.server

    def __exit__(self, exc_type, exc_value, traceback):
        print("Exiting the context")
        if exc_type:
            print(f"Exception occurred: {exc_value}")
        self.server.close()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start servers with specified model and approach.")
    parser.add_argument("--model_path", type=str, required=True, help="The path to the model.")
    parser.add_argument("--approach", type=str, default="mcts", help="The approach for the proxy server.")

    args = parser.parse_args()

    with ServerContext(VLLMServer, dict(model_path=args.model_path)) as vllm_server:
        vllm_server.wait()
        with ServerContext(ProxyServer, dict(model_path=args.model_path, approach=args.approach)) as proxy_server:
            proxy_server.wait()
            while True:
                time.sleep(5)

            # Do your stuff here
