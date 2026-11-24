from __future__ import annotations

import io
import os
import re
from typing import Any

from PIL import Image

import verl.utils.torch_functional as verl_F
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.model import compute_position_id_with_mask

from recipe.med_twi.agentic_zoomin_retrieve.qwen3_vl_4b_instruct.prompts import (
    SYSTEM_PROMPT as DEFAULT_SYSTEM_PROMPT,
)


def _extract_user_content(prompt: Any) -> str:
    if isinstance(prompt, list):
        for message in reversed(prompt):
            if isinstance(message, dict) and message.get("role") == "user":
                content = message.get("content", "")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            text_parts.append(str(item["text"]))
                    return "\n".join(text_parts).strip()
        return str(prompt[-1].get("content", "")) if prompt and isinstance(prompt[-1], dict) else ""
    if isinstance(prompt, str):
        return prompt
    return str(prompt)


def _load_image(image: Any) -> Image.Image:
    """Decode the image representations returned by different Parquet loaders."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, dict):
        image_path = image.get("path")
        if isinstance(image_path, str) and os.path.isfile(image_path):
            return Image.open(image_path).convert("RGB")
        image_bytes = image.get("bytes")
        if image_bytes is not None:
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
        raise TypeError("image dict must contain a readable `path` or raw `bytes`")
    if isinstance(image, (bytes, bytearray, memoryview)):
        return Image.open(io.BytesIO(bytes(image))).convert("RGB")
    raise TypeError(f"unsupported image type: {type(image)}")


class CustomRLHFDataset(RLHFDataset):
    system_prompt = DEFAULT_SYSTEM_PROMPT
    agent_name = "tool_agent"

    def _build_messages(self, example: dict):
        messages: list = example[self.prompt_key]
        images = example.pop(self.image_key, None) or []
        videos = example.pop(self.video_key, None) or []

        image_offset, video_offset = 0, 0
        for message in messages:
            if not images and not videos:
                continue
            assert self.processor is not None, "processor is needed to process image and video"

            content = message["content"]
            if not isinstance(content, str):
                continue

            content_list = []
            segments = re.split("(<image>|<video>)", content)
            segments = [item for item in segments if item != ""]
            for segment in segments:
                if segment == "<image>":
                    assert image_offset < len(images), f"image_offset {image_offset} >= len(images) {len(images)}"
                    image = images[image_offset]
                    if isinstance(image, Image.Image):
                        content_list.append({"type": "image", "image": image.convert("RGB")})
                    elif isinstance(image, dict):
                        image_path = image.get("path")
                        if isinstance(image_path, str) and os.path.isfile(image_path):
                            image_uri = image_path if image_path.startswith("file://") else f"file://{image_path}"
                            content_list.append({"type": "image", "image": image_uri})
                        elif "bytes" in image:
                            content_list.append({"type": "image", "image": _load_image(image)})
                        else:
                            raise TypeError("image dict must contain a readable `path` or raw `bytes`")
                    else:
                        raise TypeError(f"image must be dict or PIL.Image, unsupported image type: {type(image)}")
                    image_offset += 1
                elif segment == "<video>":
                    assert video_offset < len(videos), f"video_offset {video_offset} >= len(videos) {len(videos)}"
                    content_list.append({"type": "video", **videos[video_offset]})
                    video_offset += 1
                else:
                    content_list.append({"type": "text", "text": segment})
            message["content"] = content_list

        assert image_offset == len(images), f"image_offset {image_offset} != len(images) {len(images)}"
        assert video_offset == len(videos), f"video_offset {video_offset} != len(videos) {len(videos)}"
        return messages

    def __getitem__(self, item):
        row_dict: dict[str, Any] = self.dataframe[item]

        user_content = _extract_user_content(row_dict.get(self.prompt_key, ""))
        row_dict[self.prompt_key] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        images = []
        row_dict_images = row_dict.get(self.image_key, None)
        if row_dict_images:
            for image in row_dict_images:
                images.append(_load_image(image))

        messages = self._build_messages(row_dict)

        if self.processor is not None:
            raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.processor(text=[raw_prompt], images=images or None, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")
            model_inputs.pop("second_per_grid_ts", None)
        else:
            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids

        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt

        row_dict["index"] = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["agent_name"] = self.agent_name
        return row_dict
