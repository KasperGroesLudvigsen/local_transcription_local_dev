#from transcriber import Transcriber
import os
from pathlib import Path
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from huggingface_hub import login
import dotenv

dotenv.load_dotenv()

hf_token = os.getenv("HF_TOKEN")
print(hf_token)

login(token=hf_token)
#transcriber = Transcriber(token=hf_token)

audio_file_path = "test.wav"

test_file = Path(audio_file_path)

test_file = audio_file_path

# Sæt device og data type for optimal performance
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Angiv model ID for Hviske v3
model_id = "syvai/hviske-v3-conversation"

# Hent model og processor fra Hugging Face
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, #token=hf_token
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id) # token=hf_token

# Opret en ASR pipeline
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    torch_dtype=torch_dtype,
    device=device,
)

result = pipe(test_file)

print(result)