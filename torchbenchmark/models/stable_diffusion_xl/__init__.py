"""
HuggingFace Stable Diffusion model.
It requires users to specify "HUGGINGFACE_AUTH_TOKEN" in environment variable
to authorize login and agree HuggingFace terms and conditions.
"""
from torchbenchmark.tasks import COMPUTER_VISION
from torchbenchmark.util.model import BenchmarkModel
from torchbenchmark.util.framework.huggingface.model_factory import HuggingFaceAuthMixin

import torch
from diffusers import DiffusionPipeline


class Model(BenchmarkModel, HuggingFaceAuthMixin):
    task = COMPUTER_VISION.GENERATION

    DEFAULT_TRAIN_BSIZE = 1
    DEFAULT_EVAL_BSIZE = 1
    ALLOW_CUSTOMIZE_BSIZE = False

    def __init__(self, test, device, batch_size=None, extra_args=[]):
        HuggingFaceAuthMixin.__init__(self)
        super().__init__(test=test, device=device,
                         batch_size=batch_size, extra_args=extra_args)
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        self.pipe = DiffusionPipeline.from_pretrained(model_id).to(device)
        self.list_of_inputs = [
            torch.randn(1, 4, 256, 256).to(self.device),
            torch.tensor([1.0]).to(self.device),
            torch.randn(1, 1, 2048).to(self.device),
            {"text_embeds": torch.randn(1, 2560).to(self.device), "time_ids": torch.tensor([1]).to(self.device)}
        ]
    
    def get_module(self):
        self.pipe.unet, self.list_of_inputs

    def train(self):
        raise NotImplementedError("Train is not implemented for the stable diffusion model.")

    def eval(self):
        image = self.pipe(*self.list_of_inputs)
        return (image, )
