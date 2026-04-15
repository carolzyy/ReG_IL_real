from abc import ABC, abstractmethod

import torch
import cv2
import numpy as np
from torchvision import transforms


class CLIP():

    def __init__(
        self,
        model_class="openai/clip-vit-base-patch16",
        pooling=None,  # [None | "avg" | "max"]
        token_idx=None,  # [None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        mm_vision_select_layer=-2,
        device="cuda",
    ):

        # init model
        from transformers import CLIPVisionModel, AutoProcessor

        self.model = CLIPVisionModel.from_pretrained(model_class)
        self.model.eval()
        self.model.to(device)
        self.processor = AutoProcessor.from_pretrained(model_class)

        # model args
        self.embedding_file_key = "CLIP"
        self.pooling = pooling
        self.token_idx = token_idx
        self.mm_vision_select_layer = mm_vision_select_layer
        self.device = device

        super().__init__()

    def preprocess(self, imgs):

        inputs = self.processor(images=imgs, return_tensors="pt")

        return inputs["pixel_values"].to(self.device)

    def encode(self, imgs,output_hidden_states=False):

        postprocessed_imgs = self.preprocess(imgs)

        outputs = self.model(pixel_values=postprocessed_imgs, output_hidden_states=output_hidden_states)

        features = outputs.pooler_output

        if self.pooling is not None:
            if self.pooling == "avg":
                features = torch.mean(features, dim=1)
            elif self.pooling == "max":
                features = torch.max(features, dim=1).values

        elif self.token_idx is not None:
            features = features[:, self.token_idx]

        return features.detach().cpu().numpy().squeeze()


class Dinov3:
    def __init__(self,
                 ):
        self.dino = torch.hub.load('facebookresearch/dinov3', "dinov3_vitb16")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':
            self.dino.cuda()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    def encode(self, img):
        with torch.no_grad(), torch.inference_mode():
            img_t = self.transform(img).to(self.device)
            if img_t.ndim == 3:
                input = img_t.unsqueeze(0)#1
            features = self.dino(input)
            return features.detach().cpu().numpy().squeeze()

class SigLipv2():
    def __init__(
            self,
            model_name="google/siglip2-base-patch16-224",
            device="cuda",
            pooling=None,  # ["cls", "avg", None]
            #normalize=True,
    ):
        super().__init__()
        from transformers import Siglip2Model, AutoProcessor

        self.model = Siglip2Model.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.to(device).eval()

        self.device = device
        self.pooling = pooling
        self.normalize = False

    def preprocess(self, imgs, texts=None):
        """
        Preprocess images (and optionally texts).
        imgs: List[PIL.Image] or np.ndarray
        texts: List[str] or None
        """
        return self.processor(
            images=imgs,
            text=texts,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

    def encode(self, imgs):
        """
        Encode only images → returns embeddings.
        """
        inputs = self.preprocess(imgs, texts=None)
        outputs = self.model(**inputs)

        feats = outputs.image_embeds  # already pooled [CLS] from HuggingFace
        if self.pooling == "avg":
            feats = outputs.last_hidden_state.mean(dim=1)
        elif self.pooling is None:
            feats = outputs.last_hidden_state  # return sequence

        return feats

    def encode_all(self, imgs, texts):
        """
        Encode both images and texts (aligned pairs).
        """
        inputs = self.preprocess(imgs, texts)
        outputs = self.model(**inputs)

        img_feats = outputs.image_embeds
        txt_feats = outputs.text_embeds

        return img_feats, txt_feats



def get_encoders(model_names=None):
    """
    Returns a list of encoders to use for embedding based on the specified model_name.

    Args:
        model_name (str, optional): The name of the encoder model to use.
            Supported values: "DINOv2", "CLIP", or None for default (both).

    Returns:
        List[Encoder]: A list of encoder instances.
    """
    # Mapping from model names to encoder constructors
    encoder_registry = {
        'DINO': lambda: Dinov3(),
        #'CLIP': lambda: CLIP(),
        #'SigLip': lambda: SigLipv2()
    }
    models = {}

    if model_names is not None:
        for name in model_names:
            models[name] = encoder_registry[name]()
    else:
        for name, factory in encoder_registry.items():
            models[name] = factory()

    return models
