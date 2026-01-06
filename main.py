import os
import argparse
import warnings
from dotenv import load_dotenv

from pipelines.factory import create_pipeline
from pipelines.strategies import RUNNERS
from utils.util import ensure_dir, load_dataset, load_image, save_first_image, set_seed
from utils.GammaScheduler import GammaScheduler


def parse_args():
    parser = argparse.ArgumentParser()

    # shared
    parser.add_argument("--model", type=str, required=True,
                        choices=["lcm", "flux"])
    parser.add_argument("--dataset_config", type=str, default="./dataset.json")
    parser.add_argument("--base_image_path", type=str, default="./")
    parser.add_argument("--out_dir", type=str, default="./results")
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--num_inference", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--skip_step", type=bool, default=False,
                        help="Skips the step when gamma = 1.0 to save computational resources, though it may slightly impact performance.")

    # LCM only

    parser.add_argument("--is_p2p", type=int, default=0,
                        help="LCM only. 1 True, 0 False")
    parser.add_argument("--self_replace_steps", type=float, default=0.6)
    parser.add_argument("--cross_replace_steps", type=float, default=0.7)
    parser.add_argument("--reg_mode", type=int, default=1,
                        help="1: Approximate but precise strength control. 2: Original but imprecise strength control")
    # FLUX only
    parser.add_argument("--hf_token", type=str, default=None)

    return parser.parse_args()


def apply_defaults(args):
    if args.model == "lcm":
        if args.num_inference is None:
            args.num_inference = 15
        if args.height is None:
            args.height = 512
        if args.width is None:
            args.width = 512

    else:  # flux
        if args.num_inference is None:
            args.num_inference = 28
        if args.height is None:
            args.height = 1024
        if args.width is None:
            args.width = 1024
    return args


def warn_irrelevant_args(args):
    if args.model == "flux" and args.is_p2p == 1:
        warnings.warn("--is_p2p is ignored when --model flux")
    if args.model == "lcm" and args.hf_token is not None:
        warnings.warn("--hf_token is ignored when --model lcm")


def main():
    load_dotenv(".env")

    args = parse_args()
    args = apply_defaults(args)
    warn_irrelevant_args(args)

    hf_token = args.hf_token or os.getenv("hf_token")

    cfg = {
        "model": args.model,
        "dataset_config": args.dataset_config,
        "base_image_path": args.base_image_path,
        "out_dir": args.out_dir,
        "num_inference": args.num_inference,
        "height": args.height,
        "width": args.width,
        "skip_step": args.skip_step,
        # lcm only
        "device": args.device,
        "is_p2p": (args.is_p2p == 1),
        "self_replace_steps": args.self_replace_steps,
        "cross_replace_steps": args.cross_replace_steps,
        "reg_mode": args.reg_mode,
        # flux only
        "hf_token": hf_token,
    }

    set_seed(args.seed)
    ensure_dir(cfg["out_dir"])

    pipe = create_pipeline(
        cfg["model"],
        device=cfg["device"],
        hf_token=cfg["hf_token"],
    )

    runner = RUNNERS[cfg["model"]]
    examples = load_dataset(cfg["dataset_config"])

    if args.model == "lcm":
        k = 10 
        gamma_step = GammaScheduler.generate(
            length=cfg["num_inference"], mode="step", value=1.0, k=k)

        # k = 9 
        # custom_list =  [0.0, 0.8, 0.8, 0.8, 0.0, 0.0]
        # gamma_step = GammaScheduler.generate(
        #     length=cfg["num_inference"], mode="custom_tail", value=1.0, k=k, custom_list=custom_list)
    elif args.model == "flux":
        k = 5
        gamma_step = GammaScheduler.generate(
            length=cfg["num_inference"], mode="step", value=0.8, k=k)

    for ex in examples:
        print("key:", ex.key)
        image = load_image(cfg["base_image_path"], ex.image_path)
        ori_width, ori_height = image.size

        out = runner(pipe, image, gamma_step, ex, cfg)
        save_path = os.path.join(
            cfg["out_dir"], f"{os.path.basename(ex.image_path)}")
        save_first_image(out.images, save_path, ori_height, ori_width)


if __name__ == "__main__":
    main()
