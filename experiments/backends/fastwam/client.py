#!/usr/bin/env python
"""Client / launcher for FastWAM policy servers.

  python client.py ping   --socks 'run/sock/*.sock'
  python client.py info   --socks run/sock/g2_0.sock
  python client.py run    --socks 'run/sock/*.sock' --episodes eps.jsonl --out runs/x [--override DIR --arms orig,edit]
                          [--record rec.json] [--save-video] [--per-request 1]
  python client.py summarize --out runs/x

Episodes are sharded dynamically over all servers (work queue; one thread per server).
Results are appended to <out>/results.jsonl as they arrive (resume-safe: finished keys are skipped).
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import queue
import socket
import sys
import threading
import time
from pathlib import Path


def call(sock_path, req, timeout=None):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect(str(sock_path))
    with s:
        f = s.makefile("rwb")
        f.write((json.dumps(req) + "\n").encode())
        f.flush()
        line = f.readline()
    if not line:
        raise ConnectionError(f"empty response from {sock_path}")
    return json.loads(line)


def ep_key(e, arm=None):
    k = f"{e['suite']}|{e.get('variant_id', 'idx' + str(e.get('task_idx')))}|{e.get('seed', 7)}|{e.get('init_idx', 0)}"
    return k if arm is None else k + "|" + arm


def run_sharded(socks, episodes, out, base_req, per_request=1, log=print):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    res_path = out / "results.jsonl"
    arms = base_req.get("arms") or (["edit"] if base_req.get("weights_override") else ["orig"])
    done = set()
    if res_path.exists():
        for line in res_path.read_text().splitlines():
            r = json.loads(line)
            if not str(r.get("failure") or "").startswith("error"):
                done.add(ep_key(r, r["arm"]))
    todo = [e for e in episodes if any(ep_key(e, a) not in done for a in arms)]
    log(f"{len(episodes)} episodes, {len(todo)} to run, {len(socks)} servers")
    q = queue.Queue()
    for i in range(0, len(todo), per_request):
        q.put(todo[i : i + per_request])
    lock = threading.Lock()
    t0 = time.time()
    n_done = [0]

    def worker(sock):
        while True:
            try:
                batch = q.get_nowait()
            except queue.Empty:
                return
            req = dict(base_req, cmd="run", episodes=batch, out_dir=str(out / "artifacts"))
            try:
                resp = call(sock, req)
            except Exception as exc:
                log(f"[{sock}] transport error {exc!r}; requeue + drop server")
                q.put(batch)
                return
            if not resp.get("ok"):
                log(f"[{sock}] server error: {resp.get('error')}\n{resp.get('traceback', '')}")
                rs = [dict(e, arm=a, success=False, failure="error: " + str(resp.get("error"))) for e in batch for a in arms]
            else:
                rs = resp["results"]
            with lock:
                with res_path.open("a") as f:
                    for r in rs:
                        r["server"] = Path(sock).name
                        r["server_max_reserved_gb"] = resp.get("gpu_max_reserved_gb")
                        f.write(json.dumps(r) + "\n")
                n_done[0] += len(batch)
                log(f"{time.strftime('%H:%M:%S')} {n_done[0]}/{len(todo)} done ({time.time() - t0:.0f}s) "
                    + " ".join(f"{r['suite']}:{r['variant_id']}:{r['arm']}={'S' if r['success'] else 'F'}" for r in rs))

    ths = [threading.Thread(target=worker, args=(s,), daemon=True) for s in socks]
    [t.start() for t in ths]
    [t.join() for t in ths]
    return summarize(out)


# ------------------------------------------------------------------------------------------
# Python API (import client as C)
# ------------------------------------------------------------------------------------------
def list_socks(pattern="/mnt/SSD2_16TB/jusheng/wsp/fastwam/run/sock/*.sock"):
    return sorted(glob.glob(pattern))


def rollout(episodes, out, socks=None, weights_override=None, arms=None, override_mode="replace",
            record=None, save_frames=0, save_video=False, language="strip", per_request=1, log=print):
    """Run episodes (list of {suite, variant_id, seed[, init_idx]}) sharded over servers.
    arms: subset of ["orig","edit"] (default: ["edit"] if weights_override else ["orig"]); each episode is
    run once per arm with identical (variant, seed). Returns list of per-episode result dicts (also
    appended to <out>/results.jsonl; summary in <out>/summary.md)."""
    req = {"override_mode": override_mode, "save_frames": int(save_frames), "save_video": bool(save_video),
           "language": language}
    if weights_override:
        req["weights_override"] = str(Path(weights_override).resolve())
    if arms:
        req["arms"] = list(arms)
    if record:
        req["record"], req["compile"] = record, False
    run_sharded(socks or list_socks(), episodes, out, req, per_request, log)
    keys = {ep_key(e, a) for e in episodes for a in (arms or (["edit"] if weights_override else ["orig"]))}
    rows = {}
    for line in (Path(out) / "results.jsonl").read_text().splitlines():
        r = json.loads(line)
        if ep_key(r, r["arm"]) in keys:
            rows[ep_key(r, r["arm"])] = r
    return list(rows.values())


def probe(inputs, out, sock=None, weights_override=None, arms=None, override_mode="replace",
          record=None, attention_layers=None, compile=False):
    """Simulation-free forward on saved policy inputs (paths to *_inputs.pt written by save_frames).
    Returns per (arm, input): paths of action_norm/action .npy, activations .npz, attention maps
    ({layer: .npy [heads, L_action, L_video]}). attention_layers: <=4 layers, <=4 inputs."""
    req = {"cmd": "probe", "inputs": [str(Path(p).resolve()) for p in inputs], "out_dir": str(Path(out).resolve()), "compile": bool(compile),
           "override_mode": override_mode}
    if weights_override:
        req["weights_override"] = str(weights_override)
    if arms:
        req["arms"] = list(arms)
    if record:
        req["record"] = record
    if attention_layers:
        req["attention"] = {"layers": list(attention_layers)}
    resp = call(sock or list_socks()[0], req)
    if not resp.get("ok"):
        raise RuntimeError(resp.get("error", "") + "\n" + resp.get("traceback", ""))
    return resp


PUBLISHED = {"Camera Viewpoints": 48.4, "Robot Initial States": 75.7, "Language Instructions": 66.8,
             "Light Conditions": 96.6, "Background Textures": 68.5, "Sensor Noise": 70.1,
             "Objects Layout": 76.5, "TOTAL": 70.8}


def summarize(out):
    out = Path(out)
    rows = {}
    for line in (out / "results.jsonl").read_text().splitlines():
        r = json.loads(line)
        rows[ep_key(r, r["arm"])] = r  # last write wins
    rows = list(rows.values())
    lines = []
    for arm in sorted({r["arm"] for r in rows}):
        rs = [r for r in rows if r["arm"] == arm]
        errs = [r for r in rs if str(r.get("failure") or "").startswith("error")]
        ok = [r for r in rs if r not in errs]
        lines.append(f"## arm={arm}  episodes={len(rs)}  errors={len(errs)}")
        for key in ("category", "suite"):
            agg = collections.defaultdict(lambda: [0, 0])
            for r in ok:
                agg[r.get(key)][0] += int(r["success"])
                agg[r.get(key)][1] += 1
            lines.append(f"\n| {key} | n | success | published |\n|---|---|---|---|")
            for k in sorted(agg, key=str):
                s, n = agg[k]
                pub = PUBLISHED.get(k, "") if key == "category" else ""
                lines.append(f"| {k} | {n} | {100 * s / max(n, 1):.1f}% | {pub} |")
        s = sum(r["success"] for r in ok)
        lines.append(f"| TOTAL | {len(ok)} | {100 * s / max(len(ok), 1):.1f}% | {PUBLISHED['TOTAL']} |")
        walls = sorted(r["wall_s"] for r in ok if "wall_s" in r)
        if walls:
            lines.append(f"\nwall/episode: mean {sum(walls) / len(walls):.1f}s median {walls[len(walls) // 2]:.1f}s "
                         f"max {walls[-1]:.1f}s; mean steps {sum(r['steps'] for r in ok) / len(ok):.0f}")
        for r in errs[:5]:
            lines.append(f"ERR {r['suite']} v{r['variant_id']}: {r.get('failure')}")
    txt = "\n".join(lines)
    (out / "summary.md").write_text(txt + "\n")
    return txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ping", "info", "params", "run", "summarize"])
    ap.add_argument("--socks", default="/mnt/SSD2_16TB/jusheng/wsp/fastwam/run/sock/*.sock")
    ap.add_argument("--episodes")
    ap.add_argument("--out")
    ap.add_argument("--override")
    ap.add_argument("--override-mode", default="replace")
    ap.add_argument("--arms")
    ap.add_argument("--record", help="JSON file or inline JSON for the record spec")
    ap.add_argument("--save-video", action="store_true")
    ap.add_argument("--save-frames", type=int, default=0)
    ap.add_argument("--language", default="strip")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--per-request", type=int, default=1)
    ap.add_argument("--prefix", default="")
    a = ap.parse_args()
    socks = sorted({x for part in a.socks.split(",") for x in (glob.glob(part) if any(c in part for c in "*?[") else [part])})
    if a.cmd in ("ping", "info"):
        for s in socks:
            try:
                print(s, json.dumps(call(s, {"cmd": a.cmd}, timeout=30)))
            except Exception as exc:
                print(s, "DOWN", repr(exc))
        return
    if a.cmd == "params":
        for n, shp, dt in call(socks[0], {"cmd": "params", "prefix": a.prefix})["params"]:
            print(n, shp, dt)
        return
    if a.cmd == "summarize":
        print(summarize(a.out))
        return
    eps = [json.loads(l) for l in Path(a.episodes).read_text().splitlines() if l.strip()]
    req = {"save_video": a.save_video, "override_mode": a.override_mode, "save_frames": a.save_frames,
           "language": a.language}
    if a.override:
        req["weights_override"] = str(Path(a.override).resolve())
    if a.arms:
        req["arms"] = a.arms.split(",")
    if a.record:
        req["record"] = json.loads(Path(a.record).read_text() if Path(a.record).exists() else a.record)
        req["compile"] = False
    if a.no_compile:
        req["compile"] = False
    print(run_sharded(socks, eps, a.out, req, a.per_request, log=lambda m: print(m, flush=True)))


if __name__ == "__main__":
    sys.exit(main())
