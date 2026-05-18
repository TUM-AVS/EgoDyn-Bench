#!/usr/bin/env python3
"""
Generate Cosmos Transfer 2.5 inference spec files for chunked CARLA videos.

Reads the manifest from chunk_carla_videos.py and creates one JSON spec per
clip, plus a master list for batch inference.

Cosmos Transfer 2.5 "vis" (blur) mode is used — it takes the input RGB video,
auto-generates a blurred control signal, and produces a photorealistic output.
This is the simplest mode requiring no separate depth/edge/seg maps.

Usage:
    python scripts/prepare_cosmos_batch.py \
        --chunks-dir output/carla_chunks \
        --cosmos-dir output/cosmos_batch \
        --prompt "Dashcam footage of a vehicle driving on a road during daytime."
"""

import argparse
import json
import os
from pathlib import Path


DEFAULT_PROMPT = (
    "Dashcam footage of a car driving on a road. "
    "Photorealistic, high quality, natural lighting."
)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--chunks-dir", type=str, required=True,
        help="Directory with chunked CARLA videos and manifest.json",
    )
    parser.add_argument(
        "--cosmos-dir", type=str, default="output/cosmos_batch",
        help="Output directory for Cosmos spec files",
    )
    parser.add_argument(
        "--prompt", type=str, default=DEFAULT_PROMPT,
        help="Text prompt for Cosmos generation",
    )
    parser.add_argument(
        "--control-weight", type=float, default=1.0,
        help="Control weight for vis (blur) modality (default: 1.0)",
    )
    parser.add_argument(
        "--guidance", type=float, default=3.0,
        help="Guidance scale (default: 3.0)",
    )
    args = parser.parse_args()

    chunks_dir = os.path.abspath(args.chunks_dir)
    cosmos_dir = os.path.abspath(args.cosmos_dir)

    # Load manifest
    manifest_path = os.path.join(chunks_dir, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"Loaded manifest: {len(manifest)} clips")

    # Create output dirs
    specs_dir = os.path.join(cosmos_dir, "specs")
    prompts_dir = os.path.join(cosmos_dir, "prompts")
    output_dir = os.path.join(cosmos_dir, "outputs")
    os.makedirs(specs_dir, exist_ok=True)
    os.makedirs(prompts_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Write shared prompt file
    prompt_path = os.path.join(prompts_dir, "prompt.txt")
    with open(prompt_path, "w") as f:
        f.write(args.prompt)

    # Generate one spec per clip
    spec_paths = []
    for entry in manifest:
        clip_id = entry["clip_id"]
        video_path = os.path.join(chunks_dir, entry["input_path"])

        spec = {
            "name": clip_id,
            "prompt_path": prompt_path,
            "video_path": video_path,
            "guidance": args.guidance,
            "vis": {
                "control_weight": args.control_weight,
            },
        }

        spec_file = os.path.join(specs_dir, f"{clip_id}.json")
        with open(spec_file, "w") as f:
            json.dump(spec, f, indent=2)
        spec_paths.append(spec_file)

    # Write master list of all spec paths (for batch inference)
    master_path = os.path.join(cosmos_dir, "all_specs.txt")
    with open(master_path, "w") as f:
        for p in spec_paths:
            f.write(p + "\n")

    print(f"Generated {len(spec_paths)} spec files in {specs_dir}/")
    print(f"Master list: {master_path}")
    print(f"Prompt: {prompt_path}")
    print(f"\nTo run inference on LRZ cluster:")
    print(f"  python examples/inference.py \\")
    print(f"    -i {' '.join(spec_paths[:2])} ... \\")
    print(f"    -o {output_dir} control:vis")


if __name__ == "__main__":
    main()
