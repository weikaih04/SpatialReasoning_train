#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
Convert weikaih/multi_view_counting_training_v5 to ThinkMorph unified_edit parquet format.

Output schema per row:
  - instruction_list: [question]
  - input_image_list: [frame_0, frame_1, frame_2, frame_3]
  - output_image_list: [topdown_map]
  - output_text_list: ["<think>...<image_start>", "<image_end><answer>X</answer>"]
"""

import argparse
import io
import json
import math
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset, load_dataset_builder, Image as HFImage
from PIL import Image


def image_to_bytes(pil_image):
    if pil_image is None:
        return None
    if isinstance(pil_image, (bytes, bytearray)):
        return bytes(pil_image)
    if isinstance(pil_image, dict) and pil_image.get("bytes") is not None:
        return pil_image.get("bytes")
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format="PNG")
    return img_byte_arr.getvalue()


def has_choice_markers(text):
    if text is None:
        return False
    markers = ["A)", "A.", "(A)"]
    return any(m in text for m in markers)


def format_question(question, choices):
    question = (question or "").strip()
    choices = choices or []
    if choices and not has_choice_markers(question):
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        formatted = []
        for idx, choice in enumerate(choices):
            if idx >= len(letters):
                break
            formatted.append(f"{letters[idx]}) {choice}")
        if question:
            question = question + "\n" + "\n".join(formatted)
        else:
            question = "\n".join(formatted)
    return question


def build_output_text(answer):
    answer = (answer or "").strip()
    think_text = "<think>Use the top-down map to reason.</think><image_start>"
    answer_text = f"<image_end><answer>{answer}</answer>"
    return [think_text, answer_text]


def reformat_sample(sample):
    question = format_question(sample.get("question"), sample.get("choices"))
    answer = sample.get("answer")
    if not question or not answer:
        return None

    frames = [
        sample.get("frame_0"),
        sample.get("frame_1"),
        sample.get("frame_2"),
        sample.get("frame_3"),
    ]
    if any(frame is None for frame in frames):
        return None

    topdown = sample.get("topdown_map")
    if topdown is None:
        return None

    input_image_list = [image_to_bytes(frame) for frame in frames]
    output_image_list = [image_to_bytes(topdown)]

    if any(img is None for img in input_image_list) or output_image_list[0] is None:
        return None

    return {
        "instruction_list": [question],
        "input_image_list": input_image_list,
        "output_image_list": output_image_list,
        "output_text_list": build_output_text(answer),
    }


def write_parquet_shard(rows, parquet_file):
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, parquet_file)


def collect_existing_shards(output_dir):
    shards = []
    for path in sorted(output_dir.glob("chunk_*.parquet")):
        try:
            shard_idx = int(path.stem.split("_")[-1])
        except Exception:
            continue
        try:
            pf = pq.ParquetFile(path)
            num_rows = pf.metadata.num_rows
            num_row_groups = pf.num_row_groups
        except Exception:
            continue
        shards.append((shard_idx, path, num_rows, num_row_groups))
    return shards


def resolve_rows_per_shard(dataset_name, split, num_shards, rows_per_shard):
    if rows_per_shard is not None:
        return rows_per_shard
    if num_shards is None:
        return 2000

    try:
        builder = load_dataset_builder(dataset_name)
        split_info = builder.info.splits.get(split)
        if split_info is None or split_info.num_examples is None:
            return 2000
        return int(math.ceil(split_info.num_examples / num_shards))
    except Exception:
        return 2000


def main():
    parser = argparse.ArgumentParser(description="Convert multi-view counting dataset to parquet")
    parser.add_argument(
        "--dataset",
        type=str,
        default="weikaih/multi_view_counting_training_v5",
        help="Hugging Face dataset name",
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "data/custom_datasets/multi_view_counting/"
            "bagel_example/editing/multi_view_counting_v5"
        ),
    )
    parser.add_argument("--num-shards", type=int, default=5)
    parser.add_argument("--rows-per-shard", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--streaming", action="store_true", default=True)
    parser.add_argument("--no-streaming", dest="streaming", action="store_false")
    parser.add_argument("--raw-bytes", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    existing_shards = collect_existing_shards(output_dir) if output_dir.exists() else []
    if existing_shards and not (args.overwrite or args.resume):
        raise SystemExit(
            f"Output directory {output_dir} already has parquet files. "
            "Use --overwrite to replace or --resume to continue."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_per_shard = resolve_rows_per_shard(
        args.dataset, args.split, args.num_shards, args.rows_per_shard
    )

    print("=" * 80)
    print("Multi-view counting -> unified_edit parquet")
    print(f"Dataset: {args.dataset} ({args.split})")
    print(f"Output: {output_dir}")
    print(f"Rows per shard: {rows_per_shard}")
    print("Streaming:" , args.streaming)
    print("=" * 80)

    dataset = load_dataset(args.dataset, split=args.split, streaming=args.streaming)
    if args.raw_bytes:
        image_columns = ["topdown_map", "frame_0", "frame_1", "frame_2", "frame_3", "frame_4"]
        for col in image_columns:
            if col in dataset.features:
                dataset = dataset.cast_column(col, HFImage(decode=False))

    parquet_info = {}
    skip_converted = 0
    shard_idx = 0
    if args.resume and existing_shards:
        existing_shards.sort(key=lambda x: x[0])
        shard_idx = existing_shards[-1][0] + 1
        for _, path, num_rows, num_row_groups in existing_shards:
            parquet_info[str(path.resolve())] = {
                "num_row_groups": num_row_groups,
                "num_rows": num_rows,
            }
            skip_converted += num_rows
        print(f"Resuming: found {len(existing_shards)} shard(s), skip {skip_converted} converted samples.")
    rows = []
    num_converted = skip_converted
    num_skipped = 0
    num_skipped_while_resuming = 0

    for sample in dataset:
        record = reformat_sample(sample)
        if record is None:
            num_skipped += 1
            continue

        if skip_converted > 0:
            num_skipped_while_resuming += 1
            skip_converted -= 1
            continue

        rows.append(record)
        num_converted += 1

        if args.max_samples is not None and num_converted >= args.max_samples:
            break

        if len(rows) >= rows_per_shard:
            parquet_file = output_dir / f"chunk_{shard_idx}.parquet"
            write_parquet_shard(rows, parquet_file)
            num_rows = len(rows)
            parquet_info[str(parquet_file.resolve())] = {
                "num_row_groups": pq.ParquetFile(parquet_file).num_row_groups,
                "num_rows": num_rows,
            }
            print(f"Wrote {parquet_file} with {num_rows} rows")
            shard_idx += 1
            rows = []

    if rows:
        parquet_file = output_dir / f"chunk_{shard_idx}.parquet"
        write_parquet_shard(rows, parquet_file)
        num_rows = len(rows)
        parquet_info[str(parquet_file.resolve())] = {
            "num_row_groups": pq.ParquetFile(parquet_file).num_row_groups,
            "num_rows": num_rows,
        }
        print(f"Wrote {parquet_file} with {num_rows} rows")

    info_file = output_dir / "parquet_info.json"
    with open(info_file, "w") as f:
        json.dump(parquet_info, f, indent=2)

    print("=" * 80)
    print("Done")
    print(f"Converted: {num_converted}")
    print(f"Skipped: {num_skipped}")
    if args.resume:
        print(f"Skipped while resuming: {num_skipped_while_resuming}")
    print(f"Parquet dir: {output_dir}")
    print(f"Info file: {info_file}")
    print("=" * 80)
    print("Suggested dataset_info.py entry:")
    print("{")
    print(f"  'data_dir': '{output_dir.resolve()}',")
    print(f"  'num_files': {shard_idx + (1 if rows else 0)},")
    print(f"  'num_total_samples': {num_converted},")
    print(f"  'parquet_info_path': '{info_file.resolve()}',")
    print("}")


if __name__ == "__main__":
    main()
