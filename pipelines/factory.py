from __future__ import annotations
from typing import Optional
import torch

from .TweezeEdit_lcm import TweezeEditLCMPipeline
from .lcm_scheduler import LCMScheduler
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import CLIPTokenizer, CLIPTextModel, CLIPImageProcessor

from .TweezeEdit_flux import TweezeEditFLUXPipeline


def create_pipeline(
    model: str,
    device: str = "cuda:0",
    hf_token: Optional[str] = None,
):
    if model == "lcm":
        model_id = "SimianLuo/LCM_Dreamshaper_v7"

        vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        text_encoder = CLIPTextModel.from_pretrained(
            model_id, subfolder="text_encoder")
        tokenizer = CLIPTokenizer.from_pretrained(
            model_id, subfolder="tokenizer")
        unet = UNet2DConditionModel.from_pretrained(
            model_id,
            subfolder="unet",
            device_map=None,
            low_cpu_mem_usage=False,
            local_files_only=True,
        )
        safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            model_id, subfolder="safety_checker")
        feature_extractor = CLIPImageProcessor.from_pretrained(
            model_id, subfolder="feature_extractor")

        scheduler = LCMScheduler(
            beta_start=0.00085,
            beta_end=0.0120,
            beta_schedule="scaled_linear",
            prediction_type="epsilon",
        )

        pipe = TweezeEditLCMPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        ).to(device)

        return pipe

    if model == "flux":
        model_id = "black-forest-labs/FLUX.1-dev"
        pipe = TweezeEditFLUXPipeline.from_pretrained(
            model_id,
            token=hf_token,
            torch_dtype=torch.float16
        ).to(device)
        return pipe

    raise ValueError(f"Unknown model='{model}'. Choose from: lcm, flux")
