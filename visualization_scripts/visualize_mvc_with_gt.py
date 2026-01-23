#!/usr/bin/env python3
"""
Visualize Multi-View Counting results with ground truth thought images.
Shows: input frames, GT topdown map, model generated map, QA comparison
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


def load_hf_dataset(nsamples=None):
    """Load AI2Thor Multi-View Counting dataset from HuggingFace."""
    ds = load_dataset('weikaih/ai2thor-multiview-counting-val-800-v2-400', split='train')

    samples = []
    for i, ex in enumerate(ds):
        if nsamples is not None and i >= nsamples:
            break
        samples.append(ex)
    return samples


def visualize_sample(result_row, hf_sample, gen_img_dir, output_dir, idx, hit):
    """Visualize a single sample with input frames, GT thought, model thought, and QA."""

    # Count input frames
    frames = []
    for i in range(8):
        frame = hf_sample.get(f'frame_{i}')
        if frame is not None:
            frames.append(frame)
    num_frames = len(frames)

    # Layout: 2 rows
    # Row 1: Input frames (up to 5)
    # Row 2: GT topdown, Model generated, QA text
    fig = plt.figure(figsize=(28, 12))

    # Dynamic columns based on frame count
    ncols_top = min(num_frames, 5)
    gs = gridspec.GridSpec(2, max(ncols_top, 3), figure=fig, height_ratios=[1, 1.2])

    # Get data
    question = result_row.get('question', 'N/A')
    gt_answer = result_row.get('answer', 'N/A')
    prediction = str(result_row.get('prediction', 'N/A'))
    category = hf_sample.get('movement_type', 'unknown')
    query_object = hf_sample.get('query_object', 'unknown')
    trajectory_id = hf_sample.get('trajectory_id', 'unknown')

    choices = {
        'A': result_row.get('A', ''),
        'B': result_row.get('B', ''),
        'C': result_row.get('C', ''),
        'D': result_row.get('D', ''),
    }

    # Extract model answer
    answer_match = re.search(r'<answer>([A-D])</answer>', prediction)
    pred_answer = answer_match.group(1) if answer_match else 'N/A'

    is_correct = hit == 1
    result_str = 'CORRECT' if is_correct else 'WRONG'
    result_color = 'green' if is_correct else 'red'

    # === Row 1: Input frames ===
    for i, frame in enumerate(frames[:ncols_top]):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(frame)
        ax.set_title(f'INPUT Frame {i}', fontsize=10, fontweight='bold', color='blue')
        ax.axis('off')

    # === Row 2: GT topdown (left), Model generated (middle), Text (right) ===

    # Ground truth topdown map
    ax_gt = fig.add_subplot(gs[1, 0])
    if hf_sample and 'topdown_map' in hf_sample and hf_sample['topdown_map'] is not None:
        ax_gt.imshow(hf_sample['topdown_map'])
        ax_gt.set_title('GROUND TRUTH: Topdown Map', fontsize=12, fontweight='bold', color='purple')
    else:
        ax_gt.text(0.5, 0.5, 'GT topdown not available', ha='center', va='center')
        ax_gt.set_facecolor('#f0f0f0')
    ax_gt.axis('off')

    # Model generated image
    ax_model = fig.add_subplot(gs[1, 1])
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
        ax_model.set_title('MODEL: Generated Topdown', fontsize=12, fontweight='bold', color='green')
    else:
        ax_model.text(0.5, 0.5, 'No model image generated', ha='center', va='center', fontsize=11)
        ax_model.set_facecolor('#fff5f5')
    ax_model.axis('off')

    # === Text panel (right) ===
    ax_text = fig.add_subplot(gs[1, 2])
    ax_text.axis('off')

    # Format question (remove choices from question text for cleaner display)
    q_text = question.split('\n')[0] if '\n' in question else question
    wrapped_q = '\n'.join(textwrap.wrap(str(q_text), width=50))

    # Extract thinking process
    think_match = re.search(r'<think>(.*?)</think>', prediction, re.DOTALL)
    thinking = think_match.group(1).strip() if think_match else 'N/A'
    wrapped_think = '\n'.join(textwrap.wrap(thinking[:300], width=50))

    text_content = f"""QUESTION:
{wrapped_q}

QUERY OBJECT: {query_object}
TRAJECTORY: {trajectory_id}
CATEGORY: {category}

CHOICES:
  A: {choices['A']}    B: {choices['B']}
  C: {choices['C']}    D: {choices['D']}

MODEL'S THINKING:
{wrapped_think}

{'─' * 40}
GROUND TRUTH: {gt_answer}
MODEL ANSWER: {pred_answer}

RESULT: {result_str}
{'─' * 40}
"""

    ax_text.text(0.02, 0.98, text_content, transform=ax_text.transAxes,
                 fontsize=11, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Add large result indicator
    ax_text.text(0.5, 0.02, result_str, transform=ax_text.transAxes,
                 fontsize=28, fontweight='bold', color=result_color,
                 ha='center', va='bottom')

    plt.suptitle(f'Multi-View Counting Sample {idx} - {category} - {query_object}',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(output_dir, f'mvc_viz_{idx:04d}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path} ({result_str})')
    return save_path


def main():
    parser = argparse.ArgumentParser(description='Visualize MVC results with GT thought')
    parser.add_argument('--result_path', type=str,
                        default='VLMEvalKit_Thinkmorph/outputs_eval/thinkmorph_mvc/thinkmorph_mvc_AI2ThorMultiViewCounting_10.xlsx',
                        help='Path to results xlsx file')
    parser.add_argument('--result_pkl', type=str,
                        default=None,
                        help='Path to results pkl file with hit info (optional)')
    parser.add_argument('--gen_img_dir', type=str,
                        default='results/mvc/run_8gpu/0009880',
                        help='Directory containing generated images')
    parser.add_argument('--output_dir', type=str,
                        default='visualization_scripts/visualizations_mvc_with_gt',
                        help='Output directory for visualizations')
    parser.add_argument('--nsamples', type=int, default=10,
                        help='Number of samples to visualize')
    parser.add_argument('--category', type=str, default=None,
                        help='Filter by category (e.g., multi_camera, rotation)')
    parser.add_argument('--show_wrong', action='store_true',
                        help='Only show wrong predictions')
    parser.add_argument('--show_correct', action='store_true',
                        help='Only show correct predictions')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model results from {args.result_path}...")
    df = pd.read_excel(args.result_path)
    print(f"Loaded {len(df)} samples")

    # Try to load hit info from pkl or compute from df
    if args.result_pkl and os.path.exists(args.result_pkl):
        with open(args.result_pkl, 'rb') as f:
            results = pickle.load(f)
        df['hit'] = df.index.map(lambda x: results.get(x, {}).get('hit', 0))
    elif 'hit' in df.columns:
        pass  # Already has hit column
    else:
        # Compute hit from prediction and answer
        def compute_hit(row):
            pred = str(row.get('prediction', ''))
            answer = str(row.get('answer', ''))
            match = re.search(r'<answer>([A-D])</answer>', pred)
            pred_ans = match.group(1) if match else ''
            return 1 if pred_ans == answer else 0
        df['hit'] = df.apply(compute_hit, axis=1)

    print(f"Accuracy: {df['hit'].mean():.1%}")

    print("Loading HuggingFace dataset...")
    hf_samples = load_hf_dataset(nsamples=len(df))
    print(f"Loaded {len(hf_samples)} HF samples")

    # Filter by category if specified
    indices_to_viz = list(range(len(df)))

    if args.category:
        indices_to_viz = [i for i in indices_to_viz
                         if i < len(hf_samples) and hf_samples[i].get('movement_type') == args.category]
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
