# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import os
import time
import torch
import subprocess
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoImageProcessor, StoppingCriteria

MODEL_ID = "Salesforce/blip3-phi3-mini-instruct-r-v1"
MODEL_CACHE = "checkpoints"
MODEL_URL = "https://weights.replicate.delivery/default/salesforce/blip3-phi3-mini-instruct-r-v1/model.tar"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class EosListStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_sequence = [32007]):
        self.eos_sequence = eos_sequence

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_ids = input_ids[:,-len(self.eos_sequence):].tolist()
        return self.eos_sequence in last_ids     
    
class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)
        model = AutoModelForVision2Seq.from_pretrained(
            MODEL_CACHE,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_CACHE,
            use_fast=False,
            legacy=False
        )
        self.image_processor = AutoImageProcessor.from_pretrained(
            MODEL_CACHE,
            trust_remote_code=True
        )
        self.model = model.cuda()

    def predict(
        self,
        image: Path = Input(description="Input image"),
        question: str = Input(description="Question to ask about this image", default="how many dogs are in the picture?"),
        system_prompt: str = Input(description="System prompt", default="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions"),
        max_new_tokens: int = Input(description="Maximum number of tokens to generate", default=768, ge=512, le=2047),
    ) -> str:
        """Run a single prediction on the model"""
        t1 = time.time()
        raw_image = Image.open(image).convert('RGB')
        tokenizer = self.model.update_special_tokens(self.tokenizer)
        inputs = self.image_processor([raw_image], return_tensors="pt", image_aspect_ratio='anyres')

        template = "<|system|>\n{system}<|end|>\n<|user|>\n<image>\n{question}<|end|>\n<|assistant|>\n"
        prompt = template.format(system=system_prompt, question=question)
        
        language_inputs = tokenizer([prompt], return_tensors="pt")
        inputs.update(language_inputs)
        inputs = {name: tensor.cuda() for name, tensor in inputs.items()}
        generated_text = self.model.generate(
            **inputs,
            image_size=[raw_image.size],
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False, max_new_tokens=max_new_tokens, top_p=None, num_beams=1,
            stopping_criteria = [EosListStoppingCriteria()],
        )
        prediction = tokenizer.decode(generated_text[0], skip_special_tokens=True).split("<|end|>")[0]
        print("prediction took: ", time.time() - t1)
        return prediction
