# Taken from https://github.com/JoeyBallentine/ESRGAN
# Modified by Stax124

import io
import logging
from collections import OrderedDict
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image
from rich import print  # pylint: disable=redefined-builtin
from rich.progress import BarColumn, Progress, TaskID, TimeRemainingColumn

from .utils import dataops as ops
from .utils.architecture.RRDB import RRDBNet as ESRGAN
from .utils.architecture.SPSR import SPSRNet as SPSR
from .utils.architecture.SRVGG import SRVGGNetCompact as RealESRGANv2


class SeamlessOptions(str, Enum):
    TILE = "tile"
    MIRROR = "mirror"
    REPLICATE = "replicate"
    ALPHA_PAD = "alpha_pad"


class AlphaOptions(str, Enum):
    NO_ALPHA = "none"
    BG_DIFFERENCE = "bg_difference"
    ALPHA_SEPARATELY = "separate"
    SWAPPING = "swapping"


class Upscaler:
    def __init__(
        self,
        model: str,
        reverse: bool = False,
        skip_existing: bool = False,
        delete_input: bool = False,
        seamless: Optional[SeamlessOptions] = None,
        cpu: bool = False,
        fp16: bool = False,
        device_id: int = 0,
        cache_max_split_depth: bool = False,
        binary_alpha: bool = False,
        ternary_alpha: bool = False,
        alpha_threshold: float = 0.5,
        alpha_boundary_offset: float = 0.2,
        alpha_mode: Optional[AlphaOptions] = None,
        log: logging.Logger = logging.getLogger(),
    ) -> None:
        self.model_str = model
        self.reverse = reverse
        self.skip_existing = skip_existing
        self.delete_input = delete_input
        self.seamless: Optional[SeamlessOptions] = seamless
        self.cpu = cpu
        self.fp16 = fp16
        self.device = torch.device("cpu" if self.cpu else f"cuda:{device_id}")
        self.cache_max_split_depth = cache_max_split_depth
        self.binary_alpha = binary_alpha
        self.ternary_alpha = ternary_alpha
        self.alpha_threshold = alpha_threshold
        self.alpha_boundary_offset = alpha_boundary_offset
        self.alpha_mode = alpha_mode
        self.log = log

        self.last_model: str = ""
        self.model: Union[torch.nn.Module, ESRGAN, RealESRGANv2, SPSR]

        self.last_in_nc: int = 0
        self.last_out_nc: int = 0
        self.last_nf: int = 0
        self.last_nb: int = 0
        self.last_scale: int = 0

        if self.fp16:
            torch.set_default_tensor_type(
                torch.HalfTensor if self.cpu else torch.cuda.HalfTensor  # type: ignore
            )

    def run(self, input_img: Image.Image, scale: float = 4) -> List[Image.Image]:
        model_chain = (
            self.model_str.split("+")
            if "+" in self.model_str
            else self.model_str.split(">")
        )

        for idx, model in enumerate(model_chain):
            interpolations = (
                model.split("|") if "|" in self.model_str else model.split("&")
            )

            if len(interpolations) > 1:
                for i, interpolation in enumerate(interpolations):
                    interp_model, interp_amount = (
                        interpolation.split("@")
                        if "@" in interpolation
                        else interpolation.split(":")
                    )
                    interp_model = self.__check_model_path(interp_model)
                    interpolations[i] = f"{interp_model}@{interp_amount}"
                model_chain[idx] = "&".join(interpolations)
            else:
                model_chain[idx] = self.__check_model_path(model)

        print(
            'Model{:s}: "{:s}"'.format(
                "s" if len(model_chain) > 1 else "",
                # ", ".join([Path(x).stem for x in model_chain]),
                ", ".join([x for x in model_chain]),
            )
        )

        images: List[Image.Image] = [input_img]
        outputs: List[Image.Image] = []

        # Store the maximum split depths for each model in the chain
        split_depths = {}

        with Progress(
            # SpinnerColumn(),
            "[progress.description]{task.description}",
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
            expand=True,
        ) as progress:
            task_upscaling = progress.add_task(
                f"Upscaling {len(images)} images", total=len(images)
            )
            for idx, img in enumerate(images):
                if len(model_chain) == 1:
                    self.log.info(f"Processing {str(idx).zfill(len(str(len(images))))}")

                # read image
                # We use imdecode instead of imread to work around Unicode breakage on Windows.
                # See https://jdhao.github.io/2019/09/11/opencv_unicode_image_path/
                pil_image = img.convert("RGB")
                open_cv_image = np.array(pil_image)
                # Convert RGB to BGR
                open_cv_image = open_cv_image[:, :, ::-1].copy()
                if len(open_cv_image.shape) < 3:
                    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_GRAY2BGR)

                # Seamless modes
                if self.seamless == SeamlessOptions.TILE:
                    open_cv_image = cv2.copyMakeBorder(
                        open_cv_image, 16, 16, 16, 16, cv2.BORDER_WRAP
                    )
                elif self.seamless == SeamlessOptions.MIRROR:
                    open_cv_image = cv2.copyMakeBorder(
                        open_cv_image, 16, 16, 16, 16, cv2.BORDER_REFLECT_101
                    )
                elif self.seamless == SeamlessOptions.REPLICATE:
                    open_cv_image = cv2.copyMakeBorder(
                        open_cv_image, 16, 16, 16, 16, cv2.BORDER_REPLICATE
                    )
                elif self.seamless == SeamlessOptions.ALPHA_PAD:
                    open_cv_image = cv2.copyMakeBorder(
                        open_cv_image,
                        16,
                        16,
                        16,
                        16,
                        cv2.BORDER_CONSTANT,
                        value=[0, 0, 0, 0],
                    )
                final_scale: int = 1

                task_model_chain: Optional[TaskID] = None
                if len(model_chain) > 1:
                    task_model_chain = progress.add_task(
                        f"{str(idx).zfill(len(str(len(images))))}",
                        total=len(model_chain),
                    )
                for i, model_path in enumerate(model_chain):
                    # Load the model so we can access the scale
                    self.load_model(model_path)

                    if self.cache_max_split_depth and len(split_depths.keys()) > 0:
                        rlt, depth = ops.auto_split_upscale(
                            open_cv_image,
                            self.upscale,
                            self.last_scale,
                            max_depth=split_depths[i],
                        )
                    else:
                        rlt, depth = ops.auto_split_upscale(
                            open_cv_image, self.upscale, self.last_scale
                        )
                        split_depths[i] = depth

                    final_scale *= self.last_scale

                    # This is for model chaining
                    open_cv_image = rlt.astype("uint8")
                    if len(model_chain) > 1:
                        if task_model_chain is not None:
                            progress.advance(task_model_chain)

                if self.seamless:
                    rlt = self.crop_seamless(rlt, final_scale)  # type: ignore

                in_width, in_height = input_img.size
                rlt = cv2.resize(
                    rlt,  # type: ignore
                    (
                        int(in_width * scale),
                        int(in_height * scale),
                    ),
                    interpolation=cv2.INTER_LANCZOS4,
                )

                im_buf_arr: np.ndarray
                _is_success, im_buf_arr = cv2.imencode(".png", rlt)  # type: ignore
                buffer = im_buf_arr.tobytes()

                im_pil = Image.open(io.BytesIO(buffer))

                outputs.append(im_pil)

                progress.advance(task_upscaling)

        return outputs

    def __check_model_path(self, model_path: str) -> str:
        secondary_path = Path("data") / "upscaler" / model_path

        if Path(model_path).exists():
            return str(Path(model_path))
        elif secondary_path.exists():
            return str(secondary_path)
        else:
            raise FileNotFoundError("Could not find model path")

    # This code is a somewhat modified version of BlueAmulet's fork of ESRGAN by Xinntao
    def process(self, img: np.ndarray):
        """
        Does the processing part of ESRGAN. This method only exists because the same block of code needs to be ran twice for images with transparency.

                Parameters:
                        img (array): The image to process

                Returns:
                        rlt (array): The processed image
        """
        if img.shape[2] == 3:
            img = img[:, :, [2, 1, 0]]
        elif img.shape[2] == 4:
            img = img[:, :, [2, 1, 0, 3]]
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()  # type: ignore
        if self.fp16:
            img = img.half()  # type: ignore
        img_LR = img.unsqueeze(0)  # type: ignore
        img_LR = img_LR.to(self.device)

        output = self.model(img_LR).data.squeeze(0).float().cpu().clamp_(0, 1).numpy()
        if output.shape[0] == 3:
            output = output[[2, 1, 0], :, :]
        elif output.shape[0] == 4:
            output = output[[2, 1, 0, 3], :, :]
        output = np.transpose(output, (1, 2, 0))
        return output

    def load_model(self, model_path: str):
        if model_path != self.last_model:
            # interpolating OTF, example: 4xBox:25&4xPSNR:75
            if (":" in model_path or "@" in model_path) and (
                "&" in model_path or "|" in model_path
            ):
                interps = model_path.split("&")[:2]
                model_1 = torch.load(interps[0].split("@")[0])
                model_2 = torch.load(interps[1].split("@")[0])
                state_dict = OrderedDict()
                for k, v_1 in model_1.items():
                    v_2 = model_2[k]
                    state_dict[k] = (int(interps[0].split("@")[1]) / 100) * v_1 + (
                        int(interps[1].split("@")[1]) / 100
                    ) * v_2
            else:
                state_dict = torch.load(model_path)

            # SRVGGNet Real-ESRGAN (v2)
            if (
                "params" in state_dict.keys()
                and "body.0.weight" in state_dict["params"].keys()
            ):
                self.model = RealESRGANv2(state_dict)
                self.last_in_nc = self.model.num_in_ch
                self.last_out_nc = self.model.num_out_ch
                self.last_nf = self.model.num_feat
                self.last_nb = self.model.num_conv
                self.last_scale = self.model.scale
                self.last_model = model_path
            # SPSR (ESRGAN with lots of extra layers)
            elif "f_HR_conv1.0.weight" in state_dict:
                self.model = SPSR(state_dict)
                self.last_in_nc = self.model.in_nc
                self.last_out_nc = self.model.out_nc
                self.last_nf = self.model.num_filters
                self.last_nb = self.model.num_blocks
                self.last_scale = self.model.scale
                self.last_model = model_path
            # Regular ESRGAN, "new-arch" ESRGAN, Real-ESRGAN v1
            else:
                self.model = ESRGAN(state_dict)
                self.last_in_nc = self.model.in_nc
                self.last_out_nc = self.model.out_nc
                self.last_nf = self.model.num_filters
                self.last_nb = self.model.num_blocks
                self.last_scale = self.model.scale
                self.last_model = model_path

            del state_dict
        self.model.eval()
        for k, v in self.model.named_parameters():
            v.requires_grad = False
        self.model = self.model.to(self.device)
        self.last_model = model_path

    # This code is a somewhat modified version of BlueAmulet's fork of ESRGAN by Xinntao
    def upscale(self, img: np.ndarray) -> np.ndarray:
        """
        Upscales the image passed in with the specified model

                Parameters:
                        img: The image to upscale
                        model_path (string): The model to use

                Returns:
                        output: The processed image
        """

        img = img * 1.0 / np.iinfo(img.dtype).max

        if (
            img.ndim == 3
            and img.shape[2] == 4
            and self.last_in_nc == 3
            and self.last_out_nc == 3
        ):
            # Fill alpha with white and with black, remove the difference
            if self.alpha_mode == AlphaOptions.BG_DIFFERENCE:
                img1 = np.copy(img[:, :, :3])
                img2 = np.copy(img[:, :, :3])
                for c in range(3):
                    img1[:, :, c] *= img[:, :, 3]  # type: ignore
                    img2[:, :, c] = (img2[:, :, c] - 1) * img[:, :, 3] + 1  # type: ignore

                output1 = self.process(img1)
                output2 = self.process(img2)
                alpha = 1 - np.mean(output2 - output1, axis=2)
                output = np.dstack((output1, alpha))
                output = np.clip(output, 0, 1)
            # Upscale the alpha channel itself as its own image
            elif self.alpha_mode == AlphaOptions.ALPHA_SEPARATELY:
                img1 = np.copy(img[:, :, :3])
                img2 = cv2.merge((img[:, :, 3], img[:, :, 3], img[:, :, 3]))
                output1 = self.process(img1)
                output2 = self.process(img2)
                output = cv2.merge(
                    (
                        output1[:, :, 0],
                        output1[:, :, 1],
                        output1[:, :, 2],
                        output2[:, :, 0],
                    )
                )
            # Use the alpha channel like a regular channel
            elif self.alpha_mode == AlphaOptions.SWAPPING:
                img1 = cv2.merge((img[:, :, 0], img[:, :, 1], img[:, :, 2]))
                img2 = cv2.merge((img[:, :, 1], img[:, :, 2], img[:, :, 3]))
                output1 = self.process(img1)
                output2 = self.process(img2)
                output = cv2.merge(
                    (
                        output1[:, :, 0],
                        output1[:, :, 1],
                        output1[:, :, 2],
                        output2[:, :, 2],
                    )
                )
            # Remove alpha
            else:
                img1 = np.copy(img[:, :, :3])
                output = self.process(img1)
                output = cv2.cvtColor(output, cv2.COLOR_BGR2BGRA)

            if self.binary_alpha:
                alpha = output[:, :, 3]
                threshold = self.alpha_threshold
                _, alpha = cv2.threshold(alpha, threshold, 1, cv2.THRESH_BINARY)
                output[:, :, 3] = alpha
            elif self.ternary_alpha:
                alpha = output[:, :, 3]
                half_transparent_lower_bound = (
                    self.alpha_threshold - self.alpha_boundary_offset
                )
                half_transparent_upper_bound = (
                    self.alpha_threshold + self.alpha_boundary_offset
                )
                alpha = np.where(
                    alpha < half_transparent_lower_bound,  # type: ignore
                    0,
                    np.where(alpha <= half_transparent_upper_bound, 0.5, 1),  # type: ignore
                )
                output[:, :, 3] = alpha
        else:
            if img.ndim == 2:
                img = np.tile(
                    np.expand_dims(img, axis=2), (1, 1, min(self.last_in_nc, 3))
                )
            if img.shape[2] > self.last_in_nc:  # remove extra channels
                self.log.warning("Truncating image channels")
                img = img[:, :, : self.last_in_nc]
            # pad with solid alpha channel
            elif img.shape[2] == 3 and self.last_in_nc == 4:
                img = np.dstack((img, np.full(img.shape[:-1], 1.0)))
            output = self.process(img)

        output = (output * 255.0).round()  # type: ignore

        return output

    def crop_seamless(self, img: np.ndarray, scale: int) -> np.ndarray:
        img_height, img_width = img.shape[:2]
        y, x = 16 * scale, 16 * scale
        h, w = img_height - (32 * scale), img_width - (32 * scale)
        img = img[y : y + h, x : x + w]
        return img
