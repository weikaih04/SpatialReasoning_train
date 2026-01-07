# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import io
import random
from PIL import Image, ImageFile, PngImagePlugin

from .interleave_t2i_dataset import InterleavedBaseIterableDataset, ParquetStandardIterableDataset
from ..data_utils import pil_img2rgb


Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


class UnifiedEditIterableDataset(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):

    def parse_row(self, row):
        data = self._init_data()

        def _normalize_list(val):
            if val is None:
                return []
            if hasattr(val, "tolist"):
                val = val.tolist()
            if isinstance(val, (list, tuple)):
                return list(val)
            return [val]

        instrs = _normalize_list(row.get("instruction_list", None))
        outputs = _normalize_list(row.get("output_text_list", None))

        # Backward compatible: use input_image_list/output_image_list if provided.
        if "input_image_list" in row:
            input_images = _normalize_list(row.get("input_image_list", None))
            output_images = _normalize_list(row.get("output_image_list", None))
        else:
            images = _normalize_list(row.get("image_list", None))
            input_images = images[:1]
            output_images = images[1:]

        # Filter: skip samples with missing or invalid data
        # Use len() instead of bool check because these may be numpy arrays
        if len(instrs) == 0 or len(input_images) == 0 or len(outputs) == 0:
            print(f"Skipping sample: missing instrs/images/outputs")
            return {}

        # Validate and add all input images (conditioning only, no loss)
        input_imgs = []
        for idx, img_bytes in enumerate(input_images):
            if img_bytes is None or len(img_bytes) == 0:
                print(f"Skipping sample: input image {idx} is None or empty")
                return {}
            try:
                img = pil_img2rgb(Image.open(io.BytesIO(img_bytes)))
            except Exception as e:
                print(f"Skipping sample: failed to open input image {idx}: {e}")
                return {}
            input_imgs.append(img)

        for img in input_imgs:
            # Input images: enable_cfg=False because we must NOT dropout the input image
            data = self._add_image(
                data,
                img,
                need_loss=False,
                need_vae=True,
                need_vit=True,
                enable_cfg=False,
            )

        data = self._add_text(data, instrs[0], need_loss=False)

        for idx, out_txt in enumerate(outputs):
            # Output text should NOT be dropped - we need to compute loss on it
            data = self._add_text(data, out_txt, need_loss=True, enable_cfg=False)

            if idx < len(output_images):
                # Validate subsequent images
                if output_images[idx] is None or len(output_images[idx]) == 0:
                    print(f"Skipping sample: image {idx} is None or empty")
                    return {}
                try:
                    img = pil_img2rgb(Image.open(io.BytesIO(output_images[idx])))
                except Exception as e:
                    print(f"Skipping sample: failed to open image {idx}: {e}")
                    return {}
                # Answer images: enable_cfg=True for CFG training
                data = self._add_image(
                    data,
                    img,
                    need_loss=True,
                    need_vae=True,
                    need_vit=True,
                    enable_cfg=True,
                )

        return data
