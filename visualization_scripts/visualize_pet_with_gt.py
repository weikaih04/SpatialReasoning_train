#!/usr/bin/env python3
"""
Visualize PET results with ground truth thought images.
Shows: input image, GT thought, model thought, QA comparison
"""
import os
import re
import pickle
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import textwrap
from datasets import load_dataset
import argparse


def load_hf_dataset():
    """Load AI2ThorPerspective dataset from HuggingFace."""
    ds = load_dataset('weikaih/ai2thor-perspective-qa-800-balanced-val-v3')
    # Flatten all splits into a list with split info
    all_samples = []
    for split_name in ['distance_change_closer', 'distance_change_further',
                       'relative_position_left_left', 'relative_position_left_right',
                       'relative_position_right_left', 'relative_position_right_right']:
        if split_name in ds:
            for ex in ds[split_name]:
                ex['split_name'] = split_name
                all_samples.append(ex)
    return all_samples


def visualize_sample(result_row, hf_sample, gen_img_dir, output_dir, idx, hit):
    """Visualize a single sample with input, GT thought, model thought, and QA."""

    fig = plt.figure(figsize=(24, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1, 1.3])

    # Get data
    question = result_row.get('question', 'N/A')
    gt_answer = result_row.get('answer', 'N/A')
    prediction = str(result_row.get('prediction', 'N/A'))
    category = hf_sample.get('split_name', 'unknown') if hf_sample else 'unknown'

    choices = {
        'A': result_row.get('A', ''),
        'B': result_row.get('B', ''),
    }

    # Extract model answer
    answer_match = re.search(r'<answer>([AB])</answer>', prediction)
    pred_answer = answer_match.group(1) if answer_match else 'N/A'

    is_correct = hit == 1
    result_str = 'CORRECT' if is_correct else 'WRONG'
    result_color = 'green' if is_correct else 'red'

    # === Row 1: Input image (left), GT thought (middle) ===

    # Input image
    ax_input = fig.add_subplot(gs[0, 0])
    if hf_sample and 'marked_image_no_arrow' in hf_sample:
        ax_input.imshow(hf_sample['marked_image_no_arrow'])
        ax_input.set_title('INPUT: Perspective View', fontsize=12, fontweight='bold', color='blue')
    else:
        ax_input.text(0.5, 0.5, 'Input image not available', ha='center', va='center')
    ax_input.axis('off')

    # Ground truth thought image (new_perspective column)
    ax_gt = fig.add_subplot(gs[0, 1])
    if hf_sample and 'new_perspective' in hf_sample:
        ax_gt.imshow(hf_sample['new_perspective'])
        ax_gt.set_title('GROUND TRUTH: New Perspective', fontsize=12, fontweight='bold', color='purple')
    else:
        ax_gt.text(0.5, 0.5, 'GT thought not available', ha='center', va='center')
        ax_gt.set_facecolor('#f0f0f0')
    ax_gt.axis('off')

    # === Row 2: Model generated thought (left), empty (middle) ===

    # Model generated image
    ax_model = fig.add_subplot(gs[1, 0])
    gen_img_path = None

    # Try to find image path in prediction
    if '[Image:' in prediction:
        match = re.search(r'\[Image: ([^\]]+)\]', prediction)
        if match:
            gen_img_path = match.group(1)

    # Try standard naming pattern
    if gen_img_path is None or not os.path.exists(str(gen_img_path)):
        pattern_path = os.path.join(gen_img_dir, f'thinkmorph_out_sample{idx:04d}_1.jpg')
        if os.path.exists(pattern_path):
            gen_img_path = pattern_path

    if gen_img_path and os.path.exists(gen_img_path):
        gen_img = Image.open(gen_img_path)
        ax_model.imshow(gen_img)
        ax_model.set_title('MODEL: Generated Thought', fontsize=12, fontweight='bold', color='green')
    else:
        ax_model.text(0.5, 0.5, 'No model image generated', ha='center', va='center', fontsize=11)
        ax_model.set_facecolor('#fff5f5')
    ax_model.axis('off')

    # Comparison placeholder
    ax_compare = fig.add_subplot(gs[1, 1])
    ax_compare.axis('off')
    ax_compare.text(0.5, 0.5, f'Category:\n{category}', ha='center', va='center',
                    fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # === Text panel (right column, spanning both rows) ===
    ax_text = fig.add_subplot(gs[:, 2])
    ax_text.axis('off')

    # Format text content
    wrapped_q = '\n'.join(textwrap.wrap(str(question), width=55))

    # Extract thinking process
    think_match = re.search(r'<think>(.*?)</think>', prediction, re.DOTALL)
    thinking = think_match.group(1).strip() if think_match else 'N/A'
    wrapped_think = '\n'.join(textwrap.wrap(thinking[:400], width=55))

    text_content = f"""QUESTION:
{wrapped_q}

CHOICES:
  A: {choices['A']}
  B: {choices['B']}

MODEL'S THINKING:
{wrapped_think}

────────────────────────────────────
GROUND TRUTH: {gt_answer}
MODEL ANSWER: {pred_answer}

RESULT: {result_str}
────────────────────────────────────
"""

    ax_text.text(0.02, 0.98, text_content, transform=ax_text.transAxes,
                 fontsize=11, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Add large result indicator
    ax_text.text(0.5, 0.02, result_str, transform=ax_text.transAxes,
                 fontsize=28, fontweight='bold', color=result_color,
                 ha='center', va='bottom')

    plt.suptitle(f'PET Sample {idx} - {category}', fontsize=16, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(output_dir, f'pet_viz_{idx:04d}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path} ({result_str})')
    return save_path


def main():
    parser = argparse.ArgumentParser(description='Visualize PET results with GT thought')
    parser.add_argument('--result_path', type=str,
                        default='VLMEvalKit_Thinkmorph/outputs_eval/thinkmorph_pet/T20260113_G24011931/thinkmorph_pet_AI2ThorPerspective_NoArrow.xlsx',
                        help='Path to results xlsx file')
    parser.add_argument('--result_pkl', type=str,
                        default='VLMEvalKit_Thinkmorph/outputs_eval/thinkmorph_pet/T20260113_G24011931/thinkmorph_pet_AI2ThorPerspective_NoArrow_openai_result.pkl',
                        help='Path to results pkl file with hit info')
    parser.add_argument('--gen_img_dir', type=str,
                        default='results/pet/run_8gpu',
                        help='Directory containing generated images')
    parser.add_argument('--output_dir', type=str,
                        default='visualization_scripts/visualizations_pet_with_gt',
                        help='Output directory for visualizations')
    parser.add_argument('--nsamples', type=int, default=20,
                        help='Number of samples to visualize')
    parser.add_argument('--category', type=str, default=None,
                        help='Filter by category (e.g., distance_change_closer)')
    parser.add_argument('--show_wrong', action='store_true',
                        help='Only show wrong predictions')
    parser.add_argument('--show_correct', action='store_true',
                        help='Only show correct predictions')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model results...")
    df = pd.read_excel(args.result_path)
    with open(args.result_pkl, 'rb') as f:
        results = pickle.load(f)
    df['hit'] = df.index.map(lambda x: results.get(x, {}).get('hit', 0))
    print(f"Loaded {len(df)} samples")

    print("Loading HuggingFace dataset...")
    hf_samples = load_hf_dataset()
    print(f"Loaded {len(hf_samples)} HF samples")

    # Filter by category if specified
    indices_to_viz = list(range(len(df)))

    if args.category:
        indices_to_viz = [i for i in indices_to_viz
                         if i < len(hf_samples) and hf_samples[i].get('split_name') == args.category]
        print(f"Filtered to {len(indices_to_viz)} samples in category {args.category}")

    if args.show_wrong:
        indices_to_viz = [i for i in indices_to_viz if df.iloc[i]['hit'] == 0]
        print(f"Filtered to {len(indices_to_viz)} wrong samples")
    elif args.show_correct:
        indices_to_viz = [i for i in indices_to_viz if df.iloc[i]['hit'] == 1]
        print(f"Filtered to {len(indices_to_viz)} correct samples")

    # Visualize
    saved_files = []
    for i, idx in enumerate(indices_to_viz[:args.nsamples]):
        try:
            row = df.iloc[idx]
            hf_sample = hf_samples[idx] if idx < len(hf_samples) else None
            hit = row['hit']
            save_path = visualize_sample(row, hf_sample, args.gen_img_dir, args.output_dir, idx, hit)
            saved_files.append(save_path)
        except Exception as e:
            print(f"Error visualizing sample {idx}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nCreated {len(saved_files)} visualizations in {args.output_dir}")


if __name__ == '__main__':
    main()
