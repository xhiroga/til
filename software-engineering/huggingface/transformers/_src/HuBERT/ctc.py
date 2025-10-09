import random
from pathlib import Path
import wave

import torch
from datasets import load_dataset
from transformers import AutoProcessor, HubertForCTC


dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation").sort("id")
sample_idx = random.randrange(len(dataset))
sampling_rate = dataset.features["audio"].sampling_rate

processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")
model.eval()

audio = dataset[sample_idx]["audio"]
audio_array = audio["array"]

audio_path = Path(f"outputs/demo_{sample_idx}.wav")
audio_path.parent.mkdir(exist_ok=True)
with wave.open(audio_path.as_posix(), "wb") as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(sampling_rate)
    wav.writeframes((torch.tensor(audio_array).clamp(-1.0, 1.0) * 32767).to(torch.int16).numpy().tobytes())

inputs = processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
predicted_ids = torch.argmax(logits, dim=-1)

transcription = processor.batch_decode(predicted_ids)
hypo = transcription[0]
ref = dataset[sample_idx]["text"]

print(f"Saved audio to {audio_path}")
print(f"hypo: {hypo}")
print(f"ref : {ref}")

inputs["labels"] = processor(text=ref, return_tensors="pt").input_ids
loss = model(**inputs).loss
print(f"loss: {loss.item():.2f}")
