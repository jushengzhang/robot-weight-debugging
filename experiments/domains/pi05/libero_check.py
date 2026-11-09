"""Ordinary-task check: original LIBERO (10 tasks x 5 inits per suite), original vs edited weights, on recover servers."""
import json, os, shutil, sys
sys.path.insert(0, "/mnt/SSD_16TB/jusheng/wsp_recover")
import client as C
name, d, out = sys.argv[1], sys.argv[2], sys.argv[3]
eps = [json.loads(l) for l in open("/mnt/SSD_16TB/jusheng/wsp_pi05/samples/orig_libero5.jsonl")]
socks = [s for s in C.list_socks("/mnt/SSD_16TB/jusheng/wsp_recover/run/sock/*.sock") if os.path.exists(s[:-5] + ".ready")]
shm = "/dev/shm/wsp_eval_" + name; shutil.rmtree(shm, ignore_errors=True); shutil.copytree(d, shm)
res = C.rollout(eps, out, socks=socks, weights_override=shm, arms=["orig", "edit"], log=lambda *x: None)
o = {(r["suite"], r["variant_id"]): r["success"] for r in res if r["arm"] == "orig"}
e = {(r["suite"], r["variant_id"]): r["success"] for r in res if r["arm"] == "edit"}
ks = sorted(set(o) & set(e))
line = f"LIBERO original tasks: n={len(ks)} original {100*sum(o[k] for k in ks)/len(ks):.1f} edited {100*sum(e[k] for k in ks)/len(ks):.1f} (+{sum(e[k] and not o[k] for k in ks)}/-{sum(o[k] and not e[k] for k in ks)})"
print(line); open(out + "/summary.md", "w").write(line + "\n")
