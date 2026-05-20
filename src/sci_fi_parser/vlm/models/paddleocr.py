import logging
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from ..backend import VLMBackend

logger = logging.getLogger(__name__)

MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.5"

class PaddleOCRVL15Backend(VLMBackend):
    def __init__(self, prompt: str = "Chart Recognition:"):
        self._prompt = prompt
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16
        ).to(device).eval()
        self._processor = AutoProcessor.from_pretrained(MODEL_ID)

    def predict(self, image: Image.Image, prompt: str | None = None) -> str:
        if prompt is not None:
            logger.warning("WARNING: PaddleOCR-VL-1.5 only supports fixed task prompts; ignoring custom prompt and using %r", self._prompt)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": self._prompt},
            ],
        }]
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)
        outputs = self._model.generate(**inputs, max_new_tokens=512)
        return self._processor.decode(outputs[0][inputs["input_ids"].shape[-1]:-1])
