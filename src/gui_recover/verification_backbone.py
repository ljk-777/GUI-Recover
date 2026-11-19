"""Frozen image/text features shared with the actor; no second model allocation."""

import hashlib
import numpy as np

from .verification import pixel_features, pixel_hash


class FrozenFeatures:
    def __init__(self, actor):
        self.actor = actor
        self.images, self.texts = {}, {}
        self.image_requests = self.image_calls = self.text_requests = self.text_calls = 0

    def image(self, picture):
        self.image_requests += 1
        key = pixel_hash(picture)
        if key not in self.images:
            actor = self.actor
            inputs = actor.processor.image_processor(images=[picture],return_tensors="pt").to("cuda:0")
            with actor.torch.inference_mode():
                encoded = actor.model.visual(inputs["pixel_values"].to(actor.model.visual.dtype),
                                             grid_thw=inputs["image_grid_thw"])
                feature = encoded.float().mean(dim=0).cpu().numpy()
            self.images[key] = feature / max(float(np.linalg.norm(feature)),1e-8)
            self.image_calls += 1
        return self.images[key]

    def text(self, value):
        self.text_requests += 1
        key = hashlib.sha256(value.encode()).hexdigest()
        if key not in self.texts:
            actor = self.actor
            ids = actor.processor.tokenizer(value,return_tensors="pt",truncation=True,max_length=256).input_ids.to("cuda:0")
            with actor.torch.inference_mode():
                feature = actor.model.get_input_embeddings()(ids).float().mean(dim=1)[0].cpu().numpy()
            self.texts[key] = feature / max(float(np.linalg.norm(feature)),1e-8)
            self.text_calls += 1
        return self.texts[key]

    def encode(self, view, backend="frozen"):
        pixel = pixel_features(view)
        if backend == "pixels":
            return pixel.astype(np.float32)
        if backend != "frozen":
            raise ValueError("Unknown feature backend")
        before,after = self.image(view.before),self.image(view.after)
        text = self.text(view.goal+"\nCurrent target: "+view.target+"\nProposed value: "+(view.proposed.value or ""))
        return np.concatenate([before,after,after-before,np.abs(after-before),text,pixel]).astype(np.float32)

    def counters(self):
        return {"image_requests":self.image_requests,"actual_vision_encoder_calls":self.image_calls,
                "text_requests":self.text_requests,"actual_text_embedding_calls":self.text_calls}
