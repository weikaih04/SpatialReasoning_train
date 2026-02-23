# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from .interleave_datasets import UnifiedEditIterableDataset
from .t2i_dataset import T2IIterableDataset
from .vlm_dataset import SftJSONLIterableDataset


DATASET_REGISTRY = {
    't2i_pretrain': T2IIterableDataset,
    'vlm_sft': SftJSONLIterableDataset,
    'unified_edit': UnifiedEditIterableDataset,
}


DATASET_INFO = {
    'unified_edit': {
        # ThinkMorph training datasets (interleaved reasoning) - Parquet format
        'jigsaw_assembly': {
            'data_dir': '/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/data/training_parquet/Jigsaw_Assembly',
            'num_files': 3,
            'num_total_samples': 6000,
        },
        'spatial_navigation': {
            'data_dir': '/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/data/training_parquet/Spatial_Navigation',
            'num_files': 3,
            'num_total_samples': 6000,
        },
        'visual_search': {
            'data_dir': '/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/data/training_parquet/Visual_Search',
            'num_files': 3,
            'num_total_samples': 6990,
        },
        'chart_refocus': {
            'data_dir': '/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/data/training_parquet/Chart_Refocus',
            'num_files': 3,
            'num_total_samples': 6000,
        },
        # Path Tracing dataset from HuggingFace (linjieli222/path-tracing-2point-balanced8-16k-bagel)
        'path_tracing': {
            'data_dir': '/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/data/custom_datasets/path_tracing/bagel_example/editing/path-tracing-2point-balanced8-16k',
            'num_files': 5,
            'num_total_samples': 16606,
            'parquet_info_path': '/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/data/custom_datasets/path_tracing/bagel_example/editing/path-tracing-2point-balanced8-16k/parquet_info.json',
        },
        # Perspective Taking dataset from HuggingFace (linjieli222/perspective-2point-balanced-20k-bagel)
        'perspective': {
            'data_dir': '/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/data/custom_datasets/perspective/bagel_example/editing/perspective-balanced-20k',
            'num_files': 5,
            'num_total_samples': 20531,
            'parquet_info_path': '/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/data/custom_datasets/perspective/bagel_example/editing/perspective-balanced-20k/parquet_info.json',
        },
        # Multi-view counting dataset from HuggingFace (weikaih/multi_view_counting_training_v5)
        # Uses unified format with num_input_images=4 for multi-input support
        'multi_view_counting': {
            'data_dir': '/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/bagel_example/editing/multi_view_counting_training_v5',
            'num_files': 5,
            'num_total_samples': 17079,
            'parquet_info_path': '/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/bagel_example/editing/parquet_info/multi_view_counting_training_v5.json',
        },
        # Multi-view counting - No Thought Baseline (direct answer without visual thought)
        'mvc_no_thought': {
            'data_dir': '/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/bagel_example/editing/multi_view_counting_no_thought',
            'num_files': 5,
            'num_total_samples': 17079,
            'parquet_info_path': '/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/bagel_example/editing/multi_view_counting_no_thought/parquet_info.json',
        },
        # NOTE: For MVC latent ablations (latent4, latent16, latent32), we use the same
        # 'multi_view_counting' dataset with different output_image_transform_args in the config
        # to resize output images at training time (same approach as PET ablations)
        # AI2Thor Path Tracing MMCOT dataset (QA with visual CoT)
        'path_tracing_mmcot': {
            'data_dir': '/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/bagel_example/editing/ai2thor-path-tracing-qa-train-2point-balanced8-mmcot-16k',
            'num_files': 5,
            'num_total_samples': 16610,
            'parquet_info_path': '/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/bagel_example/editing/parquet_info/ai2thor-path-tracing-qa-train-2point-balanced8-mmcot-16k.json',
        },
        # AI2Thor Path Tracing Sideview dataset (visual CoT description only)
        'path_tracing_sideview': {
            'data_dir': '/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/bagel_example/editing/ai2thor-path-tracing-train-sideview-only',
            'num_files': 5,
            'num_total_samples': 25290,
            'parquet_info_path': '/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/bagel_example/editing/parquet_info/ai2thor-path-tracing-train-sideview-only.json',
        },
        # Path Tracing with system prompt (non-mmcot version)
        'path_tracing_with_sysprompt': {
            'data_dir': '/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/data/training/path_tracing_with_sysprompt',
            'num_files': 5,
            'num_total_samples': 16606,
            'parquet_info_path': '/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/data/training/path_tracing_with_sysprompt/parquet_info.json',
        },
        # Perspective Taking with system prompt
        'perspective_with_sysprompt': {
            'data_dir': '/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/data/training/perspective_with_sysprompt',
            'num_files': 5,
            'num_total_samples': 20531,
            'parquet_info_path': '/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/data/training/perspective_with_sysprompt/parquet_info.json',
        },
        # Perspective Taking - No Thought Baseline (empty <think></think> tags)
        'perspective_no_thought': {
            'data_dir': '/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/data/training/perspective_no_thought',
            'num_files': 5,
            'num_total_samples': 20531,
            'parquet_info_path': '/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/data/training/perspective_no_thought/parquet_info.json',
        },
        # Perspective Taking - MM CoT (detailed reasoning with image generation)
        'perspective_mmcot': {
            'data_dir': '/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/data/training/perspective_mmcot',
            'num_files': 5,
            'num_total_samples': 20531,
            'parquet_info_path': '/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/data/training/perspective_mmcot/parquet_info.json',
        },
        # Perspective Taking - Text CoT (detailed reasoning without image generation)
        'perspective_textcot': {
            'data_dir': '/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/data/training/perspective_textcot',
            'num_files': 5,
            'num_total_samples': 20531,
            'parquet_info_path': '/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/data/training/perspective_textcot/parquet_info.json',
        },
        # Habitat Perspective Taking dataset (weikaih/habitat-perspective-qa-train)
        'habitat_perspective': {
            'data_dir': '/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/data/training/habitat_perspective',
            'num_files': 5,
            'num_total_samples': 19998,
            'parquet_info_path': '/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/data/training/habitat_perspective/parquet_info.json',
        },
        # Habitat Perspective Taking v2 dataset (weikaih/habitat-perspective-qa-train-v2)
        # 6 categories: distance_closer, distance_further, position_left_left, position_left_right, position_right_left, position_right_right
        'habitat_perspective_v2': {
            'data_dir': '/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/data/training/habitat_perspective_v2',
            'num_files': 5,
            'num_total_samples': 19998,
            'parquet_info_path': '/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/data/training/habitat_perspective_v2/parquet_info.json',
        },
        # Multi-view counting - MM CoT (detailed reasoning with topdown map generation)
        'mvc_mmcot': {
            'data_dir': '/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/bagel_example/editing/mvc_mmcot',
            'num_files': 5,
            'num_total_samples': 16808,
            'parquet_info_path': '/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/bagel_example/editing/mvc_mmcot/parquet_info.json',
        },
        # Multi-view counting - Text CoT (frame-by-frame reasoning, no image generation)
        'mvc_textcot': {
            'data_dir': '/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/bagel_example/editing/mvc_textcot',
            'num_files': 5,
            'num_total_samples': 16808,
            'parquet_info_path': '/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/bagel_example/editing/mvc_textcot/parquet_info.json',
        },
        # MessyTable Multi-View Counting (leo66666/messytable train split)
        # 2-7 multi-view tabletop images, visual thought (reasoning_image_0) + MCQ answer
        'messytable': {
            'data_dir': '/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/bagel_example/editing/messytable',
            'num_files': 5,
            'num_total_samples': 1880,
            'parquet_info_path': '/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/bagel_example/editing/messytable/parquet_info.json',
        },
        # ScanNet Counting (leo66666/scannet_counting train split)
        # 5-8 multi-view indoor scene images, visual thought (reasoning_image_0) + MCQ answer
        'scannet_counting': {
            'data_dir': '/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/bagel_example/editing/scannet_counting',
            'num_files': 5,
            'num_total_samples': 540,
            'parquet_info_path': '/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/bagel_example/editing/scannet_counting/parquet_info.json',
        },
        # Real Perspective Taking dataset (MahtabBg/real_perspective_taking, ScanNet/ScanNet++)
        'real_perspective': {
            'data_dir': '/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/data/training/real_perspective',
            'num_files': 5,
            'num_total_samples': 15000,
            'parquet_info_path': '/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/data/training/real_perspective/parquet_info.json',
        },
        # BAGEL example data (for reference, not used in training)
        'seedxedit_multi': {
            'data_dir': '/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/data/bagel_example/bagel_example/editing/seedxedit_multi',
            'num_files': 10,
            'num_total_samples': 1000,
        },
    },
    'vlm_sft': {
        'llava_ov': {
            'data_dir': '/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/data/bagel_example/bagel_example/vlm/images',
            'jsonl_path': '/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/data/bagel_example/bagel_example/vlm/llava_ov_si.jsonl',
            'num_total_samples': 1000
        },
    },
    't2i_pretrain': {
        't2i': {
            'data_dir': '/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/data/bagel_example/bagel_example/t2i',
            'num_files': 10,
            'num_total_samples': 1000,
        },
    },
}
