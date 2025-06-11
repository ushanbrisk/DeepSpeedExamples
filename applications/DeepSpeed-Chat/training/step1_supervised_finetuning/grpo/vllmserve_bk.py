from trl.extras.vllm_client import VLLMClient

class Server():
    def __init__(self,
                 vllm_server_host = '0.0.0.0',
                 vllm_server_port = 8000,
                 vllm_server_timeout=120.0):
        self.vllm_server_host = vllm_server_host
        self.vllm_server_port = vllm_server_port
        self.vllm_server_timeout = vllm_server_timeout


        self.vllm_client = VLLMClient(
            self.vllm_server_host, self.vllm_server_port, connection_timeout=self.vllm_server_timeout
        )

# Example usage
if __name__ == "__main__":
    from vllm import SamplingParams

    # client = VLLMClient()
    client= Server()

    # Generate completions
    responses = client.generate(["Hello, AI!", "Tell me a joke"], n=4, max_tokens=32, sampling_params=SamplingParams())
    print("Responses:", responses)  # noqa

    # Update model weights
    # from transformers import AutoModelForCausalLM
    #
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B").to("cuda")
    # client.update_model_params(model)


