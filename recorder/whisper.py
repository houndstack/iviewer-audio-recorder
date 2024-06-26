import torch
from transformers import pipeline
from datasets import load_dataset
import numpy as np



device = "cuda:0" if torch.cuda.is_available() else "cpu"
pipe = pipeline(
  "automatic-speech-recognition",
  model="openai/whisper-medium",
  chunk_length_s=30,
  device=device,
)

ds = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation", trust_remote_code=True)
sample = ds[0]["audio"]
# for i in range(1, len(ds)):
#     combined = np.append(sample["array"], ds[i]["audio"]["array"])
#     sample["array"] = combined
#     print(combined)




print(sample)
prediction = pipe(sample.copy(), batch_size=8, return_timestamps=True, generate_kwargs={"task": "translate"})["chunks"]
print(prediction)
