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
        'multi_view_counting': {
            'data_dir': '/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/data/custom_datasets/multi_view_counting/bagel_example/editing/multi_view_counting_v5',
            'num_files': 5,
            'num_total_samples': 19957,
            'parquet_info_path': '/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/data/custom_datasets/multi_view_counting/bagel_example/editing/multi_view_counting_v5/parquet_info.json',
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
