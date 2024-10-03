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
        print(f"Starting the VLLM server {model_path=}")
        self.is_ready = False
        self.model_path = model_path
        # we have to apply the chat template ourselves due to how the binary search is run on parts of sequences
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
        print(f"Starting the ProxyServer approach {approach=}")
        self.is_ready = False
        self.model_path = model_path
        self.approach = approach
        self.return_full_response = return_full_response
        # we have to apply the chat template ourselves due to how the binary search is run on parts of sequences
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
        print("exiting the context")
        # Code to release the resources or handle exceptions
        if exc_type:
            print(f"Exception occurred: {exc_value}")
        # Return True if the exception is handled, False otherwise
        self.server.close()
        return False


if __name__ == "__main__":
    with ServerContext(VLLMServer, dict(model_path="meta-llama/Llama-3.2-1B-Instruct")) as vllm_server:
        vllm_server.wait()
        with ServerContext(
            ProxyServer, dict(model_path="meta-llama/Llama-3.2-1B-Instruct", approach="mcts")
        ) as proxy_server:
            proxy_server.wait()
            while True:
                time.sleep(5)

            # do your stuff here
