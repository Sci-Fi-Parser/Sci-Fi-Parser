from abc import ABC, abstractmethod
from PIL import Image


class VLMBackend(ABC):
    @abstractmethod
    def predict(self, image: Image.Image, prompt: str | None = None) -> str: ...
