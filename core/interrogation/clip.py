import hashlib
import logging
import math
import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from diffusers.utils import is_accelerate_available
from PIL import Image
from safetensors.numpy import load_file, save_file
from tqdm import tqdm
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.models.blip.modeling_blip import BlipForConditionalGeneration
from transformers.models.blip_2 import Blip2ForConditionalGeneration

from core.config import config
from core.files import get_full_model_path
from core.inference.functions import is_bitsandbytes_available
from core.interrogation.base_interrogator import InterrogationModel, InterrogationResult
from core.optimizations import autocast
from core.types import InterrogatorQueueEntry, Job
from core.utils import convert_to_image, download_file

logger = logging.getLogger(__name__)
CACHE_URL = "https://huggingface.co/pharma/ci-preprocess/resolve/main/"


class CLIPInterrogator(InterrogationModel):
    "internal"

    def __init__(self, device: str = "cuda", autoload: bool = False):
        super().__init__(device)

        self.device = device

        self.labels: List[LabelTable] = []
        self.cache_path = get_full_model_path("_cache", model_folder="clip", force=True)
        self.caption_model: PreTrainedModel
        self.caption_processor: AutoProcessor
        self.clip_model = None
        self.clip_preprocess = None
        self.dtype: torch.dtype = config.api.dtype

        if autoload:
            self.load()

    def generate(self, job: Job) -> InterrogationResult:
        if not isinstance(job, InterrogatorQueueEntry):
            return None  # type: ignore
        image = convert_to_image(job.data.image)
        positive = self._interrogate_positive(image=image, caption=job.data.caption)
        negative = self._interrogate_negative(image=image)
        self.memory_cleanup()
        return InterrogationResult(
            positive=[(i.strip(), 1) for i in positive.split(",")],
            negative=[(i.strip(), 1) for i in negative.split(",")],
        )

    def unload(self):
        del (
            self.labels,
            self.caption_model,
            self.caption_processor,
            self.clip_model,
            self.clip_preprocess,
        )
        self.memory_cleanup()

    def _image_to_features(self, image: Image.Image) -> torch.Tensor:
        images = self.clip_preprocess(image).unsqueeze(0).to(self.device, dtype=self.dtype)  # type: ignore
        with torch.no_grad(), autocast(dtype=self.dtype):  # type: ignore
            image_features = self.clip_model.encode_image(images)  # type: ignore
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features

    def _interrogate_positive(
        self, image: Image.Image, max_flavors: int = 32, caption: Optional[str] = None
    ) -> str:
        if caption is None:
            inputs = self.caption_processor(images=image, return_tensors="pt").to(self.device, dtype=self.dtype)  # type: ignore
            tokens = self.caption_model.generate(
                **inputs, max_new_tokens=config.interrogator.caption_max_length
            )
            caption = self.caption_processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()  # type: ignore
        image_features = self._image_to_features(image)
        merged = _merge_tables(self.labels, self)
        tops = merged.rank(image_features, max_flavors)
        return _truncate_to_fit(caption + ", " + ", ".join(tops), self.tokenize)  # type: ignore

    def _interrogate_negative(self, image: Image.Image, max_flavors: int = 32) -> str:
        def _chain(
            image_features: torch.Tensor,
            phrases: List[str],
            best_prompt: str = "",
            best_sim: float = 0,
            min_count: int = 8,
            max_count: int = 32,
            descriptor="Chaining",
            reverse: bool = False,
        ) -> str:
            def _similarity(image_features: torch.Tensor, text: str) -> float:
                text_tokens = self.tokenize([text]).to(self.device, dtype=self.dtype)
                with torch.no_grad(), autocast(dtype=self.dtype):  # type: ignore
                    text_features = self.clip_model.encode_text(text_tokens)  # type: ignore
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    similarity = text_features @ image_features.T
                return similarity[0][0].item()

            phrases = set(phrases)  # type: ignore
            if not best_prompt:
                # avoid worrying about set issues
                best_prompt = self._rank_top(
                    image_features, [f for f in phrases], reverse=reverse
                )
                best_sim = _similarity(image_features, best_prompt)
                phrases.remove(best_prompt)
            curr_prompt, curr_sim = best_prompt, best_sim

            def check(addition: str, idx: int) -> bool:
                nonlocal best_prompt, best_sim, curr_prompt, curr_sim
                prompt = curr_prompt + ", " + addition
                sim = _similarity(image_features, prompt)
                if reverse:
                    sim = -sim

                if sim > best_sim:
                    best_prompt, best_sim = prompt, sim
                if sim > curr_sim or idx < min_count:
                    curr_prompt, curr_sim = prompt, sim
                    return True
                return False

            for idx in tqdm(range(max_count), desc=descriptor):
                best = self._rank_top(
                    image_features,
                    [f"{curr_prompt, {f}}" for f in phrases],
                    reverse=reverse,
                )
                flave = best[len(curr_prompt) + 2 :]
                if not check(flave, idx):
                    break
                if self.tokenize([curr_prompt])[0][-1] != 0:
                    break
                phrases.remove(flave)
            return best_prompt

        image_features = self._image_to_features(image)
        flaves = [x for x in self.labels if x.descriptor == "flavors"][0].rank(
            image_features, config.interrogator.flavor_intermediate_count, reverse=True
        )
        flaves = (
            flaves + [x for x in self.labels if x.descriptor == "negative"][0].labels
        )
        return _chain(
            image_features,
            flaves,
            max_count=max_flavors,
            reverse=True,
            descriptor="Negative chain",
        )

    def _rank_top(
        self, image_features: torch.Tensor, text_array: List[str], reverse: bool = False
    ) -> str:
        text_tokens = self.tokenize([text for text in text_array]).to(
            self.device, dtype=self.dtype
        )
        with torch.no_grad(), autocast(dtype=self.dtype):  # type: ignore
            text_features = self.clip_model.encode_text(text_tokens)  # type: ignore
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = text_features @ image_features.T
            if reverse:
                similarity = -similarity
        return text_array[similarity.argmax().item()]

    def load(self):
        # load captioner model (BLIP)
        import open_clip

        t = config.interrogator.caption_model.split("/")[1].split("-")[0].lower()

        if t == "git":
            # Not sure this supports fp16... only time will tell :)
            self.caption_model = AutoModelForCausalLM.from_pretrained(
                config.interrogator.caption_model,
                torch_dtype=self.dtype,
                load_in_8bit=is_bitsandbytes_available(),
            )
        elif t == "blip2":
            self.caption_model = Blip2ForConditionalGeneration.from_pretrained(config.interrogator.caption_model, torch_dtype=self.dtype, load_in_8bit=is_bitsandbytes_available())  # type: ignore
        else:
            self.caption_model = BlipForConditionalGeneration.from_pretrained(config.interrogator.caption_model, torch_dtype=self.dtype, load_in_8bit=is_bitsandbytes_available())  # type: ignore
        self.caption_processor = AutoProcessor.from_pretrained(config.interrogator.caption_model, torch_dtype=self.dtype, load_in_8bit=is_bitsandbytes_available())  # type: ignore

        if config.interrogator.offload_captioner:
            if is_accelerate_available() and self.device != "cpu":
                from accelerate import cpu_offload

                logger.info("Offloading captionizer to CPU.")
                cpu_offload(self.caption_model, self.device, offload_buffers=True)  # type: ignore
                cpu_offload(self.caption_processor, self.device, offload_buffers=True)  # type: ignore
            else:
                logger.warning(
                    "Accelerate is not available. Check your installation. Skipping offload."
                )
        else:
            logger.info(
                "Loaded BLIP interrogator %s on device %s. DType: %s",
                config.interrogator.caption_model,
                self.device,
                str(self.dtype),
            )
            self.caption_model.to(self.device, dtype=self.dtype)  # type: ignore
            self.caption_processor.to(self.device, dtype=self.dtype)  # type: ignore

        # load visualizer (CLIP)
        # tf is this, black???
        (
            self.clip_model,
            _,
            self.clip_preprocess,
        ) = open_clip.create_model_and_transforms(
            config.interrogator.visualizer_model.split("/")[0],
            pretrained=config.interrogator.visualizer_model.split("/")[1],
            precision="fp16"
            if self.dtype == torch.float16 and not is_cpu(self.device)
            else "fp32",
            device=self.device,
            jit=False,
            cache_dir=str(self.cache_path),
        )
        self.clip_model.eval()  # type: ignore
        self.tokenize = open_clip.get_tokenizer(
            config.interrogator.visualizer_model.split("/")[0]
        )
        self.file_folder = get_full_model_path(
            "tokens", model_folder="clip", force=True
        )
        self.file_folder.mkdir(parents=True, exist_ok=True)

        artists = _load_list(str(self.file_folder.joinpath("artists.txt")))
        artists.extend(["by " + artist for artist in artists])
        artists.extend(["inspired by " + artist for artist in artists])

        self.labels.append(LabelTable(artists, "artists", self))
        for file in [
            x for x in os.listdir(str(self.file_folder)) if Path(x).stem != "artists"
        ]:
            self.labels.append(LabelTable(_load_list(file), Path(file).stem, self))
        self.memory_cleanup()


class LabelTable:
    "internal"

    def __init__(self, labels: List[str], descriptor: str, interrogator):
        self.ignore_on_merge = descriptor.startswith("ignore-")
        if self.ignore_on_merge:
            descriptor = descriptor.removeprefix("ignore-")

        self.descriptor = descriptor

        self.chunk_size = config.interrogator.chunk_size

        self.cache_path = get_full_model_path("_cache", model_folder="clip", force=True)

        self.device = interrogator.device
        self.dtype = interrogator.dtype
        self.embeds = []
        self.labels = labels
        self.tokenize = interrogator.tokenize

        hash = hashlib.sha1(",".join(labels).encode()).hexdigest()  # type: ignore pylint: disable=redefined-builtin
        sanitized_name = config.interrogator.caption_model.replace("/", "_").replace(
            "@", "_"
        )
        self._load_cached(descriptor, hash, sanitized_name)

        if len(self.labels) != len(self.embeds):
            self.embeds = []
            chunks = np.array_split(self.labels, max(1, len(self.labels) / self.chunk_size))  # type: ignore
            for chunk in tqdm(
                chunks, desc=f"Preprocessing {descriptor}" if descriptor else None
            ):
                text_tokens = self.tokenize(chunk).to(self.device)
                with torch.no_grad(), autocast(dtype=self.dtype):  # type: ignore
                    text_features = interrogator.clip_model.encode_text(text_tokens)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    # if no workie, put a half() before the cpu()
                    text_features = text_features.cpu().numpy()
                for i in range(text_features.shape[0]):
                    self.embeds.append(text_features[i])

            if descriptor and self.cache_path:
                self.cache_path.mkdir(parents=True, exist_ok=True)
                cache_file = self.cache_path.joinpath(
                    f"{sanitized_name}_{descriptor}.safetensors"
                )
                tensors = {
                    "embeds": np.stack(self.embeds),
                    "hash": np.array([ord(c) for c in hash], dtype=np.int8),
                }
                save_file(tensors, str(cache_file))
        if is_cpu(self.device):
            self.embeds = [e.astype(np.float32) for e in self.embeds]

    def _load_cached(
        self, descriptor: str, hash_: str, sanitized_name: str
    ) -> bool:  # pylint: disable=redefined-builtin
        cached_safetensors = self.cache_path.joinpath(
            f"{sanitized_name}_{descriptor}.safetensors"
        )
        if not cached_safetensors.exists():
            download_url = CACHE_URL + f"{sanitized_name}_{descriptor}.safetensors"
            self.cache_path.mkdir(parents=True, exist_ok=True)
            download_file(download_url, cached_safetensors)
        tensors = load_file(str(cached_safetensors))
        if "hash" in tensors and "embeds" in tensors:
            if np.array_equal(
                tensors["hash"], np.array([ord(c) for c in hash_], dtype=np.int8)
            ):
                self.embeds = tensors["embeds"]
                if len(self.embeds.shape) == 2:
                    self.embeds = [self.embeds[i] for i in range(self.embeds.shape[0])]
                return True
        return False

    def _rank(
        self,
        image_features: torch.Tensor,
        text_embeds: torch.Tensor,
        top_count: int = 1,
        reverse: bool = False,
    ) -> str:
        top_count = min(top_count, len(text_embeds))
        text_embeds = torch.stack([torch.from_numpy(t) for t in text_embeds]).to(
            self.device, dtype=self.dtype
        )
        with torch.no_grad(), autocast(dtype=self.dtype):  # type: ignore
            similarity = image_features @ text_embeds.T
            if reverse:
                similarity = -similarity
        _, top_labels = similarity.float().cpu().topk(top_count, dim=-1)
        return [top_labels[0][i].numpy() for i in range(top_count)]  # type: ignore

    def rank(
        self, image_features: torch.Tensor, top_count: int = 1, reverse: bool = False
    ) -> List[str]:
        "internal"
        if len(self.labels) <= self.chunk_size:
            tops = self._rank(image_features, self.embeds, top_count=top_count, reverse=reverse)  # type: ignore
            return [self.labels[i] for i in tops]  # type: ignore

        num_chunks = int(math.ceil(len(self.labels) / self.chunk_size))
        keep_per_chunk = int(self.chunk_size / num_chunks)

        top_labels, top_embeds = [], []
        for chunk_idx in tqdm(range(num_chunks)):
            start = chunk_idx * self.chunk_size
            stop = min(start + self.chunk_size, len(self.embeds))
            tops = self._rank(image_features, self.embeds[start:stop], top_count=keep_per_chunk, reverse=reverse)  # type: ignore
            top_labels.extend([self.labels[start + i] for i in tops])  # type: ignore
            top_embeds.extend([self.embeds[start + i] for i in tops])  # type: ignore

        tops = self._rank(image_features, top_embeds, top_count=top_count)  # type: ignore
        return [top_labels[i] for i in tops]  # type: ignore


def _merge_tables(tables: List[LabelTable], ci) -> LabelTable:
    m = LabelTable([], None, ci)  # type: ignore
    for table in tables:
        if not table.ignore_on_merge:
            m.labels.extend(table.labels)
            m.embeds.extend(table.embeds)  # type: ignore
    return m


def _truncate_to_fit(text: str, tokenize) -> str:
    parts = text.split(", ")
    new_text = parts[0]
    for part in parts[1:]:
        if tokenize([new_text + part])[0][-1] != 0:
            break
        new_text += ", " + part
    return new_text


def _load_list(data_path: str, filename: Optional[str] = None) -> List[str]:
    """Load a list of strings from a file."""
    if filename is not None:
        data_path = os.path.join(data_path, filename)
    with open(data_path, "r", encoding="utf-8", errors="replace") as f:
        items = [line.strip() for line in f.readlines()]
    return items


def is_cpu(device: Union[str, torch.device]) -> bool:
    "Return whether a device is cpu"
    return device == "cpu" or device == torch.device("cpu")
