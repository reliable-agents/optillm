import subprocess
import os
import signal
import time
import requests
from openai import OpenAI



class Server():
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
        self.output_file = open(f"vllm_server.log", "w")
 
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
                response = requests.get(f'http://0.0.0.0:8000/health')
                if response.status_code == 200:
                    print("VLLMServer is ready")
                    self.is_ready = True
                    return
            except requests.exceptions.ConnectionError:
                print("VLLMServer is not ready yet, waiting for 5 seconds")
            

    def generate(self, prompt, partial_reply=None) -> str:
        self.wait()
        
        client = OpenAI(
            base_url="http://localhost:8000/v1",
        )
        messages=[
            {"role": "user", "content": prompt}
        ]        
        if partial_reply is not None:
            messages.append({"role": "assistant", "content": partial_reply})   
                 
        completion = client.chat.completions.create(
            model=self.model_path,
            messages=messages,
            max_tokens=1024,
            extra_body={
                "continue_final_message": True,
                "add_generation_prompt": False,
                } 
            if partial_reply is not None else None
        )       
        return completion.choices[0].message.content
        
        

    


class ProxyServer(Server):
    def __init__(self, model_path="", approach="") -> None:
        print(f"Starting the ProxyServer approach {approach=}")
        self.is_ready = False
        self.model_path = model_path
        self.approach = approach
        # we have to apply the chat template ourselves due to how the binary search is run on parts of sequences
        self.output_file = open(f"proxy_server.log", "w")
 
        self.process = subprocess.Popen(
            [
               "python",
                "optillm.py",
                "--base_url=http://localhost:8000/v1",
                f"--model={self.model_path}",
                f"--approach={self.approach}",
            ],
            stdout=self.output_file,
            stderr=self.output_file,
        )
    def generate(self, prompt, partial_reply=None) -> str:
        self.wait()
        
        client = OpenAI(
            base_url="http://localhost:8080/v1",
        )
        messages=[
            {"role": "user", "content": prompt}
        ]        
        if partial_reply is not None:
            messages.append({"role": "assistant", "content": partial_reply})   
                 
        completion = client.chat.completions.create(
            model=f"{self.approach}-{self.model_path}",
            messages=messages,
            max_tokens=1024,
            # extra_body={
            #     "continue_final_message": True,
            #     "add_generation_prompt": False,
            #     } 
            # if partial_reply is not None else None
        )       
        return completion.choices[0].message.content  
          
    def wait(self):
        if self.is_ready:
            return
        while True:
            time.sleep(5)
            try:
                response = requests.get(f'http://0.0.0.0:8080/health')
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
    with ServerContext(VLLMServer, dict(model_path="Qwen/Qwen2.5-Math-1.5B-Instruct")) as vllm_server:
        vllm_server.wait()
        print(vllm_server.generate("What is the capital of France?"))
        with ServerContext(ProxyServer, dict(model_path="Qwen/Qwen2.5-Math-1.5B-Instruct", approach="mcts")) as proxy_server:
            proxy_server.wait()
            print(proxy_server.generate("What is the capital of France?"))
            
            # do your stuff here
