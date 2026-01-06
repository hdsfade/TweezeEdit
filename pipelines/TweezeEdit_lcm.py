from utils.util import set_seed
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

import torch
from diffusers import DiffusionPipeline, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel, CLIPImageProcessor
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.image_processor import VaeImageProcessor
from p2p.attention_control import register_attention_control
from typing import List, Optional, Union, Dict, Any

from diffusers import logging
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class TweezeEditLCMPipeline(DiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: None,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (
            len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor)

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        prompt_embeds: None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.
        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
        """

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(
                prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1: -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(
            dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            bs_embed * num_images_per_prompt, seq_len, -1)

        # Don't need to get uncond prompt embedding because of LCM Guided Distillation
        return prompt_embeds

    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(
                    image, output_type="pil")
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(
                    image)
            safety_checker_input = self.feature_extractor(
                feature_extractor_input, return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(
                    dtype)
            )
        return image, has_nsfw_concept

    def prepare_z0_src(self, device, image):
        batch_size = image.shape[0]
        z0_src_latents = [self.vae.config.scaling_factor * self.vae.encode(
            image[i:i+1]).latent_dist.sample() for i in range(batch_size)]
        z0_src_latents = torch.cat(z0_src_latents, dim=0).to(device)
        return z0_src_latents

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, latents=None):
        shape = (batch_size, num_channels_latents, height //
                 self.vae_scale_factor, width // self.vae_scale_factor)
        if latents is None:
            latents = torch.randn(shape, dtype=dtype).to(device)
        else:
            latents = latents.to(device)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def get_w_embedding(self, w, embedding_dim=512, dtype=torch.float32):
        """
        see https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298
        Args:
        timesteps: torch.Tensor: generate embedding vectors at these timesteps
        embedding_dim: int: dimension of the embeddings to generate
        dtype: data type of the generated embeddings
        Returns:
        embedding vectors with shape `(len(timesteps), embedding_dim)`
        """
        assert len(w.shape) == 1
        w = w * 1000.

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    @torch.no_grad()
    def __call__(
        self,
        image,
        controller,
        seed: int = 42,
        is_p2p: bool = False,
        regularization_mode: int = 0,  # 0: approx; 1: simple; 2: original
        regularization_gammas: List[float] = None,
        skip_step: bool = False,
        src_prompt: Union[str, List[str]] = None,
        tar_prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = [""],
        height: Optional[int] = 768,
        width: Optional[int] = 768,
        src_guidance_scale: float = 1.5,
        tar_guidance_scale: float = 1.5,
        num_images_per_prompt: Optional[int] = 1,
        num_inference_steps: int = 4,
        blend_steps: int = 1,
        lcm_origin_steps: int = 50,
        src_prompt_embeds: Optional[torch.FloatTensor] = None,
        tar_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        set_seed(seed)

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 2. Define call parameters
        if src_prompt is not None and isinstance(src_prompt, str):
            batch_size = 1
        elif src_prompt is not None and isinstance(src_prompt, list):
            batch_size = len(src_prompt)
        else:
            batch_size = src_prompt_embeds.shape[0]

        device = self._execution_device
        # 3. Encode input prompt
        negative_prompt_embeds = self._encode_prompt(
            negative_prompt,
            device,
            num_images_per_prompt,
            prompt_embeds=None
        )
        src_prompt_embeds = self._encode_prompt(
            src_prompt,
            device,
            num_images_per_prompt,
            prompt_embeds=src_prompt_embeds,
        )
        tar_prompt_embeds = self._encode_prompt(
            tar_prompt,
            device,
            num_images_per_prompt,
            prompt_embeds=tar_prompt_embeds,
        )

        src_tar_prompt_embeds = torch.cat([negative_prompt_embeds, negative_prompt_embeds, src_prompt_embeds, tar_prompt_embeds],
                                          dim=0)
        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, lcm_origin_steps)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variable
        image = self.resize_and_preprocess_image(image, height, width, device)
        z0_src = self.prepare_z0_src(device, image)

        bs = batch_size * num_images_per_prompt

        if is_p2p:
            register_attention_control(self, controller)
        z_inv = z0_src.clone()

        if regularization_gammas is None:
            regularization_gammas = [0] * num_inference_steps

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                regularization_gamma = regularization_gammas[i]
                if regularization_gamma == 1.0 and skip_step:
                    progress_bar.update()
                    continue

                z_src, _ = self.scheduler.forward(i, t, z0_src)
                z_tar = z_inv - z0_src + z_src

                ts = t.expand(bs*4).to(device=device)

                z_src_tar_model_input = torch.cat(
                    [z_src, z_tar,  z_src, z_tar])
                model_pred = self.unet(
                    z_src_tar_model_input,
                    ts,
                    encoder_hidden_states=src_tar_prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False)[0]

                model_pred_src_uncond, model_pred_tar_uncond, model_pred_src_text, model_pred_tar_text = model_pred.chunk(
                    4, dim=0)
                model_pred_src = model_pred_src_uncond + src_guidance_scale * \
                    (model_pred_src_text - model_pred_src_uncond)
                model_pred_tar = model_pred_tar_uncond + tar_guidance_scale * \
                    (model_pred_tar_text - model_pred_tar_uncond)

                # compute the clean sample x_t -> x_0
                model_pred = torch.cat([model_pred_src, model_pred_tar])
                step_input = torch.cat([z_src, z_tar])
                _, denoised = self.scheduler.step(
                    model_pred, i, t, step_input, return_dict=False)

                denoised_src, denoised_tar = denoised.chunk(2, dim=0)

                noise_diff = (model_pred_tar - model_pred_src)
                if i < blend_steps and is_p2p:
                    denoised_src, denoised_tar = controller.step_callback(
                        torch.cat([denoised_src, denoised_tar])).chunk(2)

                z_inv = self.scheduler.step_tweezeedit(
                    z0_src, denoised_src, denoised_tar, i, t)
                if regularization_mode == 1:  # approx
                    r_t = self.scheduler.approx_rt(
                        regularization_gamma, noise_diff, denoised_src, denoised_tar, i, t)

                    z_inv = z_inv - r_t
                elif regularization_mode == 2:  # original
                    r_t = self.scheduler.original_rt(
                        regularization_gamma, noise_diff, denoised_src, denoised_tar, i, t)

                    z_inv = z_inv - r_t

                progress_bar.update()
                if i == len(timesteps) - 1:

                    z_inv = denoised_tar

        if not output_type == "latent":
            image = self.vae.decode(
                z_inv / self.vae.config.scaling_factor, return_dict=False)[0]
        else:
            image = z_inv

        has_nsfw_concept = None
        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(
            image, output_type=output_type, do_denormalize=do_denormalize)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

    def resize_and_preprocess_image(self, image, height, width, device):
        ratio = min(height / image.height, width / image.width)
        image = image.resize(
            (int(image.width * ratio), int(image.height * ratio)))
        image = self.image_processor.preprocess(image)
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        image = image.to(device)
        return image
