Earlier rollouts suggested that failures had a weaker video-expert signal in channel 1938, but different trajectories could explain that correlation. I will inspect the existing identical-observation probes to see whether the difference is already present before the robot moves.
```python
def json_structure(path, max_chars=1400):
    x = json.loads(Path(path).read_text())
    if isinstance(x, dict):
        print(Path(path).name, "keys:", list(x)[:25])
        print("preview:", json.dumps(x)[:max_chars])
    else:
        print(Path(path).name, "length:", len(x))
        print("first:", json.dumps(x[0])[:max_chars] if x else None)
    return x

natural_pairs = json_structure(work / "natural_probe_pairs.json", 1700)
compact_manifest = json_structure(work / "compact_probe_manifest.json", 1000)
sink_diagnosis = json_structure(work / "video_sink_diagnosis.json", 1800)
templates_summary = json_structure(work / "rank_one_template_summary.json", 1100)

print("Probe 4 files:", [p.name for p in (work / "probe_4").iterdir()][:12])
if (work / "probe_4" / "index.json").exists():
    probe4_index = json_structure(work / "probe_4" / "index.json", 1300)
one_frames_dir = next(p for p in (work / "roll_3_frames").iterdir() if p.is_dir())
print("Example saved episode files:", [p.name for p in one_frames_dir.iterdir()][:15])
```
