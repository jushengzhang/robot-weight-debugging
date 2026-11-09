I will inspect two same-task success/failure pairs at matching times, alongside the saved input structure. These controls reduce the risk of mistaking different instructions or gripper commands for a failure-specific visual feature.
```python
from PIL import Image, ImageDraw
from IPython.display import display

p2 = json.loads((W / "probe_2" / "results.json").read_text())
p19 = json.loads((W / "probe_19" / "results.json").read_text())
p27 = json.loads((W / "probe_27" / "results.json").read_text())

print("Probe27 observations:")
for r in p27:
    print(r["idx"], Path(r["input"]).parent.name, Path(r["input"]).name)

p = local_path(p2[0]["input"])
print("Local input:", p, "exists", p.exists())
if p.exists():
    inp = torch.load(p, map_location="cpu", weights_only=False)
    def structure(x, depth=0):
        if isinstance(x, torch.Tensor): return (tuple(x.shape), str(x.dtype))
        if isinstance(x, np.ndarray): return (x.shape, str(x.dtype))
        if isinstance(x, dict): return {k:structure(v,depth+1) for k,v in x.items()}
        return str(x)[:160]
    print("Saved input structure:", structure(inp))

z = np.load(W / "probe_19" / "probe_orig_0_acts.npz")
print("Activation groups:", Counter("::".join(k.split("::")[:2]) for k in z.files))
print("Examples:", [(k,z[k].shape) for k in z.files[:4]])
print("Direction keys:", list(torch.load(W / "analysis_directions.pt", weights_only=False).keys()))

tiles = []
for r in p2[:4]:
    frame = local_path(r["input"]).with_name(Path(r["input"]).name.replace("_inputs.pt", ".png"))
    if frame.exists():
        im = Image.open(frame).convert("RGB")
        tile = Image.new("RGB", (im.width, im.height + 26), "white")
        tile.paste(im, (0,26))
        ImageDraw.Draw(tile).text((5,5), f"Probe2 input {r['idx']}: {frame.parent.name[-30:]}", fill="black")
        tiles.append(tile)
if tiles:
    sheet = Image.new("RGB", (max(t.width for t in tiles)*2, max(t.height for t in tiles)*2), "white")
    for i,t in enumerate(tiles):
        sheet.paste(t, ((i%2)*t.width, (i//2)*t.height))
    sheet.save(W / "diagnostic_frames.png")
    display(sheet)
```
