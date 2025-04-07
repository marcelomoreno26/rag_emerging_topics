import uvicorn
import torch
from argparse import ArgumentParser
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from json import JSONDecodeError
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from huggingface_hub import hf_hub_download
from transformers import LogitsProcessor
from transformers.generation.logits_process import _calc_banned_ngram_tokens



app = FastAPI()



def supports_input_field(tokenizer) -> bool:
    """
    Check if the given tokenizer supports an input field by applying a chat template
    and verifying if the provided input content is included in the generated message.

    Parameters
    ----------
    tokenizer : object
        A tokenizer object that supports the `apply_chat_template` method.

    Returns
    -------
    bool
        True if the chat template supports the input field in the processed message,
        otherwise False.
    """
    chat = [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "input", "content": "The inauguration of Donald Trump as the reelected 47th president of the United States replacing Joe Biden after beating Kamala Harris in the elections took place on Monday, January 20, 2025."},
      {"role": "user", "content": "Who is the president of the United States?"}
    ]
    try:
      message =  tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
      if chat[1]["content"] not in message:
        return False
      return True
    except Exception as e:
      return False
    

def replace_input_field(message: list) -> list[dict]:
    """
    Replace the "input" role in a chat message with a "user" role, appending the input content
    as context to the user's message, but only if an "input" role exists.

    Parameters
    ----------
    message : list
        A list of dictionaries representing a chat conversation. The second element is expected
        to have a role of "input" to be replaced.

    Returns
    -------
    list
        A modified chat message list where the "input" role is replaced by a "user" role,
        appending the input content as context. If no replacement is needed, the original
        message list is returned.
    """
    if any(entry["role"] == "input" for entry in message):
        if message[1]["role"] == "input":
            fixed_message = [
                message[0],
                {"role": "user", "content": f"Contexto:{message[1]['content']}\n\nPregunta:{message[2]['content']}"}
            ]
            return fixed_message
    
    return message




DEFAULT_PARAMS = {
            "max_tokens":2048, 
            "top_p":0.9, 
            "temperature":0.2, 
            "min_p":0.1,
            "repetition_penalty":1.05
        }


@app.get("/api_active/")
async def is_api_active():
    return {"active": True}


@app.post("/generate/")
async def generate_prediction(request: Request):
    """
    Handles text generation requests.

    Parameters
    ----------
    request : Request
        Incoming HTTP request containing:
        - "messages" (list): Chat messages for generation.
        - "sampling_params" (dict, optional): Parameters for text generation.

    Returns
    -------
    dict
        JSON response with success status and generated predictions or an error message.
    """
    try:
        data = await request.json()
        messages = data.get("messages")
        if not messages:
            return JSONResponse(content={"success": False, "error": "Missing 'messages' key."}, status_code=400)
        
        if not supports_input_field(tokenizer):
            messages = list(map(replace_input_field, messages))
            
        messages = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        if data.get("sampling_params"):
            sampling_params = SamplingParams(**data.get("sampling_params"))
        else:
            sampling_params = SamplingParams(**DEFAULT_PARAMS)

        outputs = model.generate(messages, sampling_params=sampling_params)
        
        if not outputs:
            return JSONResponse(content={"success": False, "error": "No output from model."}, status_code=500)

        predictions = [output.outputs[0].text for output in outputs if output.outputs]

        return {"success": True, "predictions": predictions}

    except JSONDecodeError:
        return JSONResponse(content={"success": False, "error": "Invalid JSON in request."}, status_code=400)

    except KeyError as e:
        return JSONResponse(content={"success": False, "error": f"Missing key: {str(e)}."}, status_code=400)

    except RuntimeError as e:
        return JSONResponse(content={"success": False, "error": f"Text generation error: {str(e)}."}, status_code=500)

    except Exception as e:
        return JSONResponse(content={"success": False, "error": f"Unexpected error: {str(e)}."}, status_code=500)




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Hugging Face model name or path")
    parser.add_argument("--tokenizer", type=str, required=False, help="Hugging Face model name or path for the tokenizer in case it is different than the model used.")
    parser.add_argument("--filename", type=str, required=False, help="Filename of the GGUF file.")
    parser.add_argument("--port", type=int, required=False, default=8000, help="Port in which to run the API")
    parser.add_argument("--max_model_len", type=int, required=False, default=32000, help="Max model token length.")
    parser.add_argument("--gpu_memory_utilization", type=float, required=False, default=0.9, help="GPU memory utilization with vLLM. Useful if multiple models are needed to balance load.")

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer if args.tokenizer else args.model_name)
    
    if args.filename:
        model = hf_hub_download(args.model_name, filename=args.filename)
        model = LLM(model, max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization)
    else:
        model = LLM(args.model_name, max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
    #uv run llm_api.py --model_name=QuantFactory/leniachat-qwen2-1.5B-v0-GGUF --tokenizer=LenguajeNaturalAI/leniachat-qwen2-1.5B-v0 --filename=leniachat-qwen2-1.5B-v0.Q4_K_M.gguf --gpu_memory_utilization=0.85 --max_model_len=20000
    #uv run llm_api.py --model_name=Qwen/Qwen2.5-1.5B-Instruct-GGUF --tokenizer=Qwen/Qwen2.5-1.5B-Instruct --filename=qwen2.5-1.5b-instruct-q4_k_m.gguf --gpu_memory_utilization=0.85 --max_model_len=20000