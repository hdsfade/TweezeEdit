from __future__ import annotations
import torch

from utils.schemas import Example

from p2p.attention_control import LocalBlend, AttentionRefine


def run_lcm(pipe, image, gamma_step, ex: Example, cfg: dict):
    def build_controller():
        if not cfg["is_p2p"]:
            return None
        if ex.blended_word is None:
            local_blend = None
        else:
            local_blend = LocalBlend(
                [ex.src_prompt, ex.tar_prompt],
                ex.blended_word,
                device=cfg["device"],
                num_ddim_steps=cfg["num_inference"],
            )
        return AttentionRefine(
            [ex.src_prompt, ex.tar_prompt],
            cfg["num_inference"],
            cross_replace_steps=cfg["cross_replace_steps"],
            self_replace_steps=cfg["self_replace_steps"],
            local_blend=local_blend,
            device=cfg["device"],
        )

    controller = build_controller()
    with torch.no_grad():
        out = pipe(
            image=image,
            controller=controller,
            is_p2p=cfg["is_p2p"],
            regularization_mode=cfg["reg_mode"],
            regularization_gammas=gamma_step,
            skip_step=cfg["skip_step"],
            src_prompt=ex.src_prompt,
            tar_prompt=ex.tar_prompt,
            height=cfg["height"],
            width=cfg["width"],
            num_images_per_prompt=1,
            num_inference_steps=cfg["num_inference"],
            lcm_origin_steps=50,
        )

    return out


def run_flux(pipe, image, gamma_step, ex: Example, cfg: dict):
    with torch.no_grad():
        out = pipe(
            image=image,
            regularization_gammas=gamma_step,
            skip_step=cfg["skip_step"],
            src_prompt=ex.src_prompt,
            tar_prompt=ex.tar_prompt,
            height=cfg["height"],
            width=cfg["width"],
            num_inference_steps=cfg["num_inference"],
        )
    return out


RUNNERS = {
    "lcm": run_lcm,
    "flux": run_flux,
}
