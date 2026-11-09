# Figures and simulator replay

The two MP4 files show an actual selected paired simulator episode from language
editing session C. They are not generated robot imagery.

| Field | Value |
| --- | --- |
| Benchmark suite | LIBERO Goal |
| Variant / seed / initial-state index | 1174 / 7 / 0 |
| Instruction | `position the glass container of fermented grapes atop the storage unit for dishes and utensils` |
| Original policy | Timeout, 300 steps |
| Edited policy | Success, 85 steps |
| Source records | [Episode provenance](../evidence/pi05-video-provenance.jsonl) |

The GIF places the two released videos side by side, preserving both camera
views in each video. The source videos contain 10 frames per second and two
policy steps per frame. The preview samples every other frame and plays at
5 frames per second, holding the shorter rollout at its actual terminal frame.
The terminal outcome labels come from the episode records. The PNG is a static
terminal-frame preview. The selected successful repair is illustrative, not a
random sample or a measurement of average performance.

To rebuild, install Pillow and run from the repository root:

```bash
python3 -m pip install Pillow
python3 scripts/build_readme_demo.py --ffmpeg /path/to/ffmpeg
```

The script writes the GIF, PNG, and [a source-hash manifest](language-repair-preview.json).
No simulator is run and no visual content is synthesized. The other SVG files
are figures from the paper.
