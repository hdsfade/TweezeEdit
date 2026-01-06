import torch
from diffusers import FluxPipeline
from diffusers.pipelines.flux import FluxPipelineOutput
from typing import List, Optional, Union
import numpy as np
import inspect


def scale_noise(
    scheduler,
    sample: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    noise: Optional[torch.FloatTensor] = None,
) -> torch.FloatTensor:
    """
    Foward process in flow-matching

    Args:
        sample (`torch.FloatTensor`):
            The input sample.
        timestep (`int`, *optional*):
            The current timestep in the diffusion chain.

    Returns:
        `torch.FloatTensor`:
            A scaled input sample.
    """
    # if scheduler.step_index is None:
    scheduler._init_step_index(timestep)

    if noise is None:
        noise = torch.randn_like(sample).to(sample.device)

    sigma = scheduler.sigmas[scheduler.step_index]
    sample = sigma * noise + (1.0 - sigma) * sample

    return sample


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(
            scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class TweezeEditFLUXPipeline(FluxPipeline):
    @torch.no_grad()
    def resize_and_preprocess_image(self, image, height, width, device):
        ratio = min(height / image.height, width / image.width)
        image = image.resize(
            (int(image.width * ratio), int(image.height * ratio)))
        image = self.image_processor.preprocess(image)
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        image = image.to(device).half()
        return image

    def prepare_z0_src(self, device, image):
        z0_src = (self.vae.encode(image).latent_dist.mode() -
                  self.vae.config.shift_factor) * self.vae.config.scaling_factor
        return z0_src.to(device)

    def forward_noise(self, latents, t, noise=None):
        if noise is None:
            noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, t)
        return noisy_latents

    @torch.no_grad()
    def __call__(
        self,
        image,
        regularization_gammas: List[float] = None,
        skip_step: bool = False,
        src_prompt: Union[str, List[str]] = None,
        tar_prompt: Union[str, List[str]] = None,
        height: Optional[int] = 1024,
        width: Optional[int] = 1024,
        src_guidance_scale: float = 1.5,
        tar_guidance_scale: float = 5.5,
        num_inference_steps: int = 28
    ):
        device = self._execution_device
        image = self.resize_and_preprocess_image(image, height, width, device)
        z0_src = self.prepare_z0_src(device, image)
        batch_size = z0_src.shape[0]

        num_channels_latents = self.transformer.config.in_channels // 4
        z0_src, latent_image_ids = self.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            z0_src.dtype,
            device,
            None,
            z0_src,
        )
        z0_src = self._pack_latents(
            z0_src, z0_src.shape[0], num_channels_latents, z0_src.shape[2], z0_src.shape[3])

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)

        image_seq_len = z0_src.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )

        self._num_timesteps = len(timesteps)

        (
            src_prompt_embeds,
            src_pooled_prompt_embeds,
            src_text_ids,

        ) = self.encode_prompt(
            prompt=src_prompt,
            prompt_2=None,
            device=device,
        )
        self._guidance_scale = tar_guidance_scale
        (
            tar_prompt_embeds,
            tar_pooled_prompt_embeds,
            tar_text_ids,
        ) = self.encode_prompt(
            prompt=tar_prompt,
            prompt_2=None,
            device=device,
        )

        # handle guidance
        if self.transformer.config.guidance_embeds:
            src_guidance = torch.tensor([src_guidance_scale], device=device)
            src_guidance = src_guidance.expand(z0_src.shape[0])
            tar_guidance = torch.tensor([tar_guidance_scale], device=device)
            tar_guidance = tar_guidance.expand(z0_src.shape[0])
        else:
            src_guidance = None
            tar_guidance = None

        z_inv = z0_src.clone()
        if regularization_gammas == None:
            regularization_gammas = [0] * num_inference_steps

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                self.scheduler._init_step_index(t)
                sigma_idx = self.scheduler.step_index
                current_sigma = self.scheduler.sigmas[sigma_idx]
                next_sigma = self.scheduler.sigmas[sigma_idx+1]

                regularization_gamma = regularization_gammas[i]
                if regularization_gamma == 1.0 and skip_step:
                    progress_bar.update()
                    continue
                z_src = scale_noise(self.scheduler, z0_src, t)
                z_tar = z_inv + z_src - z0_src

                timestep = t.expand(z_src.shape[0]).to(z_src.dtype)
                v_src = self.transformer(
                    hidden_states=z_src,
                    timestep=timestep / 1000,
                    guidance=src_guidance,
                    pooled_projections=src_pooled_prompt_embeds,
                    encoder_hidden_states=src_prompt_embeds,
                    txt_ids=src_text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]
                v_tar = self.transformer(
                    hidden_states=z_tar,
                    timestep=timestep / 1000,
                    guidance=tar_guidance,
                    pooled_projections=tar_pooled_prompt_embeds,
                    encoder_hidden_states=tar_prompt_embeds,
                    txt_ids=tar_text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]
                dt = next_sigma - current_sigma

                z_tar = z_tar.to(torch.float32)
                z_src = z_src.to(torch.float32)

                """
                # If the sampler supports stochastic sampling, a better form can be used.
                denoised_tar = z_tar  - current_sigma * v_tar
                noise_tar = z_tar + (1.0 - current_sigma) * v_tar
                denoised_src = z_src - current_sigma * v_src
                noise_src = z_src + (1 - current_sigma) * v_src
                z_inv = z0_src + (1.0 - next_sigma)* (denoised_tar - denoised_src)
                rt = (-current_sigma + next_sigma)* (denoised_tar - denoised_src) -  next_sigma * (noise_tar - noise_src) 
                """

                z_inv = z0_src + (z_tar + dt * v_tar) - (z_src + dt * v_src)
                rt = regularization_gamma * dt * (v_tar - v_src)
                z_inv = z_inv - rt
                z_inv = z_inv.to(v_src.dtype)

                progress_bar.update()

        latents = self._unpack_latents(
            z_inv, height, width, self.vae_scale_factor)
        latents = (latents / self.vae.config.scaling_factor) + \
            self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type="pil")
        return FluxPipelineOutput(images=image)
