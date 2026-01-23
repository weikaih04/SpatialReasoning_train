#!/usr/bin/env python3
"""
Visualize Sideview model outputs: input image, GT intermediate thought, generated image, and predictions.
"""
import os
import io
import re
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import textwrap
import pyarrow.parquet as pq
import argparse


def load_results(file_path):
    """Load model output results from xlsx or pkl file."""
    if file_path.endswith('.pkl'):
        df = pd.read_pickle(file_path)
    else:
        df = pd.read_excel(file_path)
    return df


def load_sideview_parquet_sample(parquet_dir, sample_idx):
    """Load input and ground truth images from sideview parquet data."""
    parquet_files = sorted([f for f in os.listdir(parquet_dir) if f.endswith('.parquet')])

    samples_seen = 0
    for pq_file in parquet_files:
        pq_path = os.path.join(parquet_dir, pq_file)
        table = pq.read_table(pq_path)
        df_pq = table.to_pandas()

        if samples_seen + len(df_pq) > sample_idx:
            # Sample is in this file
            local_idx = sample_idx - samples_seen
            row = df_pq.iloc[local_idx]

            image_list = row['image_list']
            input_img = Image.open(io.BytesIO(image_list[0])) if len(image_list) > 0 else None
            gt_img = Image.open(io.BytesIO(image_list[1])) if len(image_list) > 1 else None

            instruction = row['instruction_list'][0] if len(row['instruction_list']) > 0 else ""
            output_texts = row['output_text_list']

            return {
                'input_image': input_img,
                'gt_sideview_image': gt_img,
                'instruction': instruction,
                'output_texts': output_texts,
            }

        samples_seen += len(df_pq)

    return None


def visualize_sample(result_row, parquet_data, gen_img_dir, output_dir, idx):
    """Visualize a single sample with input, GT sideview, generated sideview, and text."""

    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1.2, 1], width_ratios=[1, 1, 1.2])

    # Get data
    prediction = result_row.get('prediction', 'N/A')
    gt_answer = result_row.get('answer', 'N/A')

    # Row 0: Input image (topdown view)
    ax_input = fig.add_subplot(gs[0, 0])
    if parquet_data and parquet_data['input_image']:
        ax_input.imshow(parquet_data['input_image'])
        ax_input.set_title('INPUT: Top-down View', fontsize=14, fontweight='bold', color='blue')
    else:
        ax_input.text(0.5, 0.5, 'Input image not available', ha='center', va='center')
        ax_input.set_facecolor('#f5f5f5')
    ax_input.axis('off')

    # Row 0: Ground truth sideview image
    ax_gt = fig.add_subplot(gs[0, 1])
    if parquet_data and parquet_data['gt_sideview_image']:
        ax_gt.imshow(parquet_data['gt_sideview_image'])
        ax_gt.set_title('GROUND TRUTH: Intermediate Thought', fontsize=14, fontweight='bold', color='purple')
    else:
        ax_gt.text(0.5, 0.5, 'GT sideview not available', ha='center', va='center')
        ax_gt.set_facecolor('#f5f5f5')
    ax_gt.axis('off')

    # Row 1: Generated sideview image
    ax_gen = fig.add_subplot(gs[1, 0:2])
    gen_img_path = None

    # Try to find image path in prediction
    if '[Image:' in str(prediction):
        match = re.search(r'\[Image: ([^\]]+)\]', str(prediction))
        if match:
            gen_img_path = match.group(1)

    # If not found, try standard naming pattern
    if gen_img_path is None or not os.path.exists(gen_img_path):
        pattern_path = os.path.join(gen_img_dir, f'thinkmorph_out_sample{idx:04d}_1.jpg')
        if os.path.exists(pattern_path):
            gen_img_path = pattern_path

    if gen_img_path and os.path.exists(gen_img_path):
        gen_img = Image.open(gen_img_path)
        ax_gen.imshow(gen_img)
        ax_gen.set_title('MODEL GENERATED: Intermediate Thought', fontsize=14, fontweight='bold', color='green')
    else:
        ax_gen.text(0.5, 0.5, f'No generated image found\nSearched: {gen_img_path}',
                   ha='center', va='center', fontsize=12)
        ax_gen.set_facecolor('#f5f5f5')
    ax_gen.axis('off')

    # Right column: Text panel (spanning both rows)
    ax_text = fig.add_subplot(gs[:, 2])
    ax_text.axis('off')

    # Extract question (remove system prompt for display)
    question = result_row.get('question', parquet_data.get('instruction', '') if parquet_data else '')
    # Try to extract just the user question part
    if 'You are following the path' in question:
        q_match = re.search(r'(You are following.*?)$', question, re.DOTALL)
        if q_match:
            question = q_match.group(1)

    wrapped_q = '\n'.join(textwrap.wrap(str(question)[:500], width=55))

    # Extract thinking process from prediction
    think_match = re.search(r'<think>(.*?)</think>', str(prediction), re.DOTALL)
    thinking = think_match.group(1).strip() if think_match else ''

    # If no <think> tags, show raw prediction (truncated)
    if not thinking:
        # Try to get text after Round_0:
        if 'Round_0:' in str(prediction):
            thinking = str(prediction).split('Round_0:')[1][:800]
        else:
            thinking = str(prediction)[:800]

    wrapped_think = '\n'.join(textwrap.wrap(thinking[:800], width=55))

    # Ground truth text
    wrapped_gt = '\n'.join(textwrap.wrap(str(gt_answer)[:300], width=55))

    text_content = f"""QUESTION:
{wrapped_q}

GT TEXT OUTPUT:
{wrapped_gt}

MODEL'S THINKING:
{wrapped_think}

(GT image shown top-center)
(Generated image shown bottom-left)
"""

    ax_text.text(0.02, 0.98, text_content, transform=ax_text.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle(f'Sideview Sample {idx}: Intermediate Thought Visualization', fontsize=16, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(output_dir, f'sideview_viz_{idx:04d}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')
    return save_path


def main():
    parser = argparse.ArgumentParser(description='Visualize Sideview results with GT intermediate thoughts')
    parser.add_argument('--result_path', type=str,
                        default='VLMEvalKit_Thinkmorph/outputs_sideview_overfit/thinkmorph_sideview/T20260115_G24011931/thinkmorph_sideview_SideviewOverfit.xlsx',
                        help='Path to results xlsx file')
    parser.add_argument('--gen_img_dir', type=str,
                        default='results/sideview/run_8gpu/0004560_overfit',
                        help='Directory containing generated images')
    parser.add_argument('--parquet_dir', type=str,
                        default='/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/bagel_example/editing/ai2thor-path-tracing-train-sideview-only',
                        help='Directory containing sideview parquet files')
    parser.add_argument('--output_dir', type=str,
                        default='visualization_scripts/visualizations_sideview',
                        help='Output directory for visualizations')
    parser.add_argument('--nsamples', type=int, default=10,
                        help='Number of samples to visualize')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model results from {args.result_path}...")
    df = load_results(args.result_path)
    print(f"Loaded {len(df)} samples")

    saved_files = []
    for idx in range(min(args.nsamples, len(df))):
        try:
            row = df.iloc[idx]

            # Load parquet data for ground truth
            print(f"Loading parquet data for sample {idx}...")
            parquet_data = load_sideview_parquet_sample(args.parquet_dir, idx)

            save_path = visualize_sample(row, parquet_data, args.gen_img_dir, args.output_dir, idx)
            saved_files.append(save_path)
        except Exception as e:
            print(f"Error visualizing sample {idx}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nCreated {len(saved_files)} visualizations in {args.output_dir}")


if __name__ == '__main__':
    main()
