import argparse

import torch as t
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="interactive terminal session with vLLM models")
    parser.add_argument(
        "--model", 
        type=str, 
        default="/scratch/models/gemma-2-2b-it",
        help="model name or path to load"
    )
    parser.add_argument(
        "--max-tokens", 
        type=int, 
        default=2048,
        help="maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7,
        help="sampling temperature"
    )
    parser.add_argument(
        "--top-p", 
        type=float, 
        default=0.9,
        help="top-p sampling parameter"
    )
    parser.add_argument(
        "--gpu-memory-utilization", 
        type=float, 
        default=0.9,
        help="gpu memory utilization target (0.0 to 1.0)"
    )
    parser.add_argument(
        "--tensor-parallel-size", 
        type=int, 
        default=t.cuda.device_count(),
        help="number of gpus to use for tensor parallelism"
    )
    return parser.parse_args()


class ChatSession:
    def __init__(
        self, 
        model: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        
        print(f"loading model: {model}")
        self.llm = LLM(
            model=model,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
        )
        
        self.history = []
        
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
    
    def format_prompt(self) -> str:
        """format the conversation history into a prompt for the model."""
        messages = self.history.copy()
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        ) 
        return formatted_prompt
    
    def chat(self, user_input: str):
        """process user input and generate a response."""
        # add user message to history
        self.history.append({"role": "user", "content": user_input})
        
        # format the prompt
        prompt = self.format_prompt()
        
        # generate the full response
        outputs = self.llm.generate(prompt, self.sampling_params, use_tqdm=False)
        response_text = outputs[0].outputs[0].text
        print(f"Assistant: {response_text}")
        print()
        
        # add assistant response to history
        self.history.append({"role": "assistant", "content": response_text})
        
        return response_text


def main():
    args = parse_args()
    
    # initialize chat session
    session = ChatSession(
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size
    )
    
    print(f"interactive session with {args.model}")
    print("type 'exit', 'quit', or press Ctrl+D to end the session")
    print("=" * 50)
    
    try:
        while True:
            try:
                user_input = input("User: ")
                if user_input.lower() in ["exit", "quit"]:
                    break
                session.chat(user_input)
            except KeyboardInterrupt:
                print("\nuse Ctrl+D or type 'exit' to exit")
                continue
    except EOFError:
        pass
    
    print("\nending session.")


if __name__ == "__main__":
    main()
