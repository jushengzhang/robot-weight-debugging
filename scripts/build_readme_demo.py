#!/usr/bin/env python3
"""Compose the README replay from the two released simulator videos.

Requires Pillow and ffmpeg. No simulation, interpolation, or image generation is
performed. Both videos are sampled at the same rate; the shorter one is held at
its final frame. Run from any directory with --ffmpeg /path/to/ffmpeg if needed.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument(
        "--font", default="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    assets = root / "assets"
    filenames = [
        "libero_goal_v1174_s7_i0_orig_fail.mp4",
        "libero_goal_v1174_s7_i0_edit_succ.mp4",
    ]
    records = [json.loads(line) for line in
               (root / "evidence/pi05-video-provenance.jsonl").read_text().splitlines()]
    records = {r["arm"]: r for r in records if r["suite"] == "libero_goal"
               and r["variant_id"] == 1174 and r["seed"] == 7 and r["init_idx"] == 0}
    width, height, stride = 448, 224, 2
    videos = []
    for filename in filenames:
        raw = subprocess.check_output([
            args.ffmpeg, "-v", "error", "-i", str(assets / filename),
            "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
        ])
        frame_bytes = width * height * 3
        if len(raw) % frame_bytes:
            raise ValueError("Unexpected frame dimensions")
        videos.append([Image.frombytes("RGB", (width, height), raw[i:i + frame_bytes])
                       for i in range(0, len(raw), frame_bytes)])

    font = ImageFont.truetype(args.font, 21)
    small = ImageFont.truetype(args.font, 18)
    canvas_size = (952, 294)
    frames = []
    # Source videos: 10 fps and two policy steps per frame. Sample every other
    # frame for a 5 fps preview without changing the playback time scale.
    for index in range(0, max(map(len, videos)), stride):
        frame = Image.new("RGB", canvas_size, "white")
        draw = ImageDraw.Draw(frame)
        for arm, x, label, source in zip(
            ("orig", "edit"), (16, 488),
            ("Original policy", "65 embedding rows edited"), videos,
        ):
            draw.text((x, 5), label, fill="#24292f", font=font)
            frame.paste(source[min(index, len(source) - 1)], (x, 37))
            record = records[arm]
            if index >= len(source) - 1:
                outcome = "Success" if record["success"] else "Timeout"
                text = f"{outcome} at step {record['steps']}"
            else:
                text = f"Step {index * 2}"
            draw.text((x, 269), text, fill="#57606a", font=small)
        frames.append(frame)
    # Include each arm's true terminal frame, even when stride skips it.
    terminal = Image.new("RGB", canvas_size, "white")
    draw = ImageDraw.Draw(terminal)
    for arm, x, label, source in zip(
        ("orig", "edit"), (16, 488),
        ("Original policy", "65 embedding rows edited"), videos,
    ):
        draw.text((x, 5), label, fill="#24292f", font=font)
        terminal.paste(source[-1], (x, 37))
        outcome = "Success" if records[arm]["success"] else "Timeout"
        draw.text((x, 269), f"{outcome} at step {records[arm]['steps']}",
                  fill="#57606a", font=small)
    frames.append(terminal)
    terminal.save(assets / "language-repair-poster.png")
    palette_sheet = Image.new("RGB", (canvas_size[0], canvas_size[1] * 8), "white")
    for i in range(8):
        palette_sheet.paste(frames[i * (len(frames) - 1) // 7], (0, i * canvas_size[1]))
    palette = palette_sheet.quantize(colors=256)
    indexed = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    indexed[0].save(
        assets / "language-repair.gif", save_all=True, append_images=indexed[1:],
        duration=[200] * (len(indexed) - 1) + [1800], loop=0, optimize=False,
        disposal=1,
    )
    manifest = {
        "suite": "libero_goal", "variant_id": 1174, "seed": 7, "init_idx": 0,
        "instruction": records["orig"]["instruction"],
        "selection": "Illustrative original-failure / edited-success pair; not a random sample",
        "source_video_fps": 10, "source_policy_steps_per_frame": 2,
        "preview_fps": 5, "shorter_video": "Held at its actual terminal frame",
        "original_steps": records["orig"]["steps"],
        "edited_steps": records["edit"]["steps"],
        "sources": {name: hashlib.sha256((assets / name).read_bytes()).hexdigest()
                    for name in filenames},
    }
    (assets / "language-repair-preview.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Wrote language-repair.gif, language-repair-poster.png, language-repair-preview.json")


if __name__ == "__main__":
    main()
