#!/usr/bin/env python
"""[LIBERO-PRO COPY of the backend client.py: episode key = e["key"]; summary = ptype x suite vs published LIBERO-PRO]
Client / launcher for pi0.5 (openpi pi05_libero, PyTorch) LIBERO-Plus policy servers (same protocol as wsp/fastwam).

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
    if "key" in e:  # LIBERO-PRO episode
        k = f"{e['key']}|{e.get('seed', 7)}"
        return k if arm is None else k + "|" + arm
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
def list_socks(pattern="/mnt/SSD_16TB/jusheng/wsp_pro/run/sock/*.sock"):
    return sorted(glob.glob(pattern))


def rollout(episodes, out, socks=None, weights_override=None, arms=None, override_mode="replace",
            record=None, save_frames=0, save_video=False, language="strip", per_request=1, log=print,
            compile=False):
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
        req["record"] = record
    if compile:
        req["compile"] = True
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
    Returns per (arm, input): action .npy [10,7], matches_saved_chunk, activations .npz, attention maps
    ({expert layer: .npy [8 heads, 10 action tokens, L_prefix]}). attention_layers: <=4 layers, <=4 inputs."""
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


# no official pi0.5 row in the LIBERO-Plus repo/paper (arXiv 2510.13626); pi0 (openpi pi0_libero) Table 2 for reference
PUBLISHED = {"Camera Viewpoints": 13.8, "Robot Initial States": 6.0, "Language Instructions": 58.8,
             "Light Conditions": 85.0, "Background Textures": 81.4, "Sensor Noise": 79.0,
             "Objects Layout": 68.9, "TOTAL": 53.6}


PRO_PUBLISHED = {  # LIBERO-PRO README leaderboard (repo @eafdb80; arXiv 2510.03827): (pi0.5, pi0) success rates
    ("libero_goal", "object"): (0.97, 0.94), ("libero_goal", "position"): (0.38, 0.00), ("libero_goal", "semantic"): (0.97, 0.93),
    ("libero_goal", "task"): (0.00, 0.00), ("libero_goal", "environment"): (0.46, 0.39),
    ("libero_spatial", "object"): (0.97, 0.95), ("libero_spatial", "position"): (0.20, 0.00), ("libero_spatial", "semantic"): (0.97, 0.97),
    ("libero_spatial", "task"): (0.01, 0.00), ("libero_spatial", "environment"): (0.46, 0.60),
    ("libero_10", "object"): (0.92, 0.79), ("libero_10", "position"): (0.08, 0.00), ("libero_10", "semantic"): (0.93, 0.82),
    ("libero_10", "task"): (0.01, 0.00), ("libero_10", "environment"): (0.46, 0.27),
    ("libero_object", "object"): (0.98, 0.94), ("libero_object", "position"): (0.17, 0.00), ("libero_object", "semantic"): (0.96, 0.90),
    ("libero_object", "task"): (0.01, 0.00), ("libero_object", "environment"): (0.73, 0.29),
}
PRO_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
PRO_TYPES = ["original", "object", "position", "semantic", "task", "environment"]


def summarize_pro(rows):
    """markdown: success per ptype x suite (n) with published pi0.5 / pi0 in brackets"""
    import collections
    agg = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        for k in ((r["ptype"], r["suite"]), (r["ptype"], "ALL"), ("ALL", r["suite"]), ("ALL", "ALL")):
            agg[k][0] += int(r["success"])
            agg[k][1] += 1
    out = ["| ptype | " + " | ".join(PRO_SUITES) + " | all | published pi0.5 / pi0 (mean) |",
           "|---|" + "---|" * (len(PRO_SUITES) + 2)]
    for p in [x for x in PRO_TYPES if (x, "ALL") in agg] + ["ALL"]:
        cells = []
        for su in PRO_SUITES + ["ALL"]:
            sc, n = agg.get((p, su), (0, 0))
            pub = PRO_PUBLISHED.get((su, p))
            cells.append(f"{100 * sc / n:.0f}% ({n})" + (f" [{100 * pub[0]:.0f}/{100 * pub[1]:.0f}]" if pub else "") if n else "-")
        pubs = [PRO_PUBLISHED[(su, p)] for su in PRO_SUITES if (su, p) in PRO_PUBLISHED]
        pm = f"{100 * sum(x[0] for x in pubs) / len(pubs):.1f} / {100 * sum(x[1] for x in pubs) / len(pubs):.1f}" if pubs else ""
        out.append(f"| {p} | " + " | ".join(cells) + f" | {pm} |")
    return "\n".join(out)


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
        if ok and all("ptype" in r for r in ok):  # LIBERO-PRO
            lines.append("\n" + summarize_pro(ok))
            s = sum(r["success"] for r in ok)
            lines.append(f"\nTOTAL {s}/{len(ok)} = {100 * s / len(ok):.1f}%")
            walls = sorted(r["wall_s"] for r in ok if "wall_s" in r)
            lines.append(f"wall/episode: mean {sum(walls) / len(walls):.1f}s median {walls[len(walls) // 2]:.1f}s max {walls[-1]:.1f}s")
            for r in errs[:5]:
                lines.append(f"ERR {r.get('key')}: {r.get('failure')}")
            continue
        for key in ("category", "suite"):
            agg = collections.defaultdict(lambda: [0, 0])
            for r in ok:
                agg[r.get(key)][0] += int(r["success"])
                agg[r.get(key)][1] += 1
            lines.append(f"\n| {key} | n | success | published pi0 |\n|---|---|---|---|")
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
    ap.add_argument("--socks", default="/mnt/SSD_16TB/jusheng/wsp_pro/run/sock/*.sock")
    ap.add_argument("--episodes")
    ap.add_argument("--out")
    ap.add_argument("--override")
    ap.add_argument("--override-mode", default="replace")
    ap.add_argument("--arms")
    ap.add_argument("--record", help="JSON file or inline JSON for the record spec")
    ap.add_argument("--save-video", action="store_true")
    ap.add_argument("--save-frames", type=int, default=0)
    ap.add_argument("--language", default="strip")
    ap.add_argument("--compile", action="store_true", help="torch.compile sample_actions (default eager)")
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
    if a.compile:
        req["compile"] = True
    print(run_sharded(socks, eps, a.out, req, a.per_request, log=lambda m: print(m, flush=True)))


if __name__ == "__main__":
    sys.exit(main())
