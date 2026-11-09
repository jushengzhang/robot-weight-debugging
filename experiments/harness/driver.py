"""Weight-space-programming driver: an LLM edits a network's weights through a Python kernel and a forward-pass tool.

Each turn the editor replies with exactly one ```python block (run in the persistent kernel), one ```forward block
(JSON request to the domain's forward-pass tool, which spends budget), or no block. The run continues until the
forward budget is spent, then the editor gives its final answer. Every turn is logged under <run>/turns/tNNN/;
state.json allows resuming after a crash (a resumed run starts a new conversation with the handoff message).

usage: python driver.py --run <dir> --domain <module.py> [--model gpt-6-astra --effort max --passes 50]
The domain module defines: TASK (str, the first message; may use {passes} and {max_python}), PRELUDE (kernel code),
KERNEL_PYTHON, setup(run_dir) and forward(request: dict, run_dir, k) -> str (raises ValueError for a bad request).
"""
import argparse, importlib.util, json, os, re, sys, time, traceback
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kernel import Kernel

API = "https://api.openai.com/v1/responses"
BLOCK = re.compile(r"```(python|forward)[ \t]*\n(.*?)```", re.S)
EXPLAIN = ("EXPLAIN EVERY STEP: before each code or forward block, explain in plain English what you are about to do "
           "and why (one short paragraph).")


def call_llm(model, effort, inp, prev_id, log):
    body = {"model": model, "reasoning": {"effort": effort, "summary": "auto"}, "input": inp, "store": True}
    if prev_id:
        body["previous_response_id"] = prev_id
    key = os.environ["OPENAI_API_KEY"]
    delay = 10
    for attempt in range(1, 200):
        try:
            req = urllib.request.Request(API, data=json.dumps(body).encode(), method="POST",
                                         headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=3600) as r:
                d = json.load(r)
            if d.get("status") != "completed":
                raise RuntimeError(f"status {d.get('status')}: {d.get('error') or d.get('incomplete_details')}")
            text = "".join(c.get("text", "") for o in d.get("output", []) if o.get("type") == "message"
                           for c in o.get("content", []))
            summ = "\n".join(s.get("text", "") for o in d.get("output", []) if o.get("type") == "reasoning"
                             for s in o.get("summary", []))
            return d["id"], text, summ, d.get("usage", {})
        except Exception as e:  # network drops through the relay are common: retry with backoff
            msg = str(e)
            if hasattr(e, "read"):
                try:
                    msg += " " + e.read().decode()[:500]
                except Exception:
                    pass
            log(f"LLM call failed (attempt {attempt}): {msg[:300]}; retry in {delay}s")
            if "invalid_api_key" in msg or "model_not_found" in msg:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 300)
    raise RuntimeError("LLM unreachable")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--model", default="gpt-6-astra")
    ap.add_argument("--effort", default="max")
    ap.add_argument("--passes", type=int, default=50)
    ap.add_argument("--max-python", type=int, default=300)
    ap.add_argument("--handoff-every", type=int, default=40)
    ap.add_argument("--max-turns", type=int, default=600)
    a = ap.parse_args()

    spec = importlib.util.spec_from_file_location("domain", a.domain)
    dom = importlib.util.module_from_spec(spec); spec.loader.exec_module(dom)
    R = os.path.abspath(a.run); os.makedirs(R + "/turns", exist_ok=True)
    work = R + "/work"; os.makedirs(work, exist_ok=True)
    logf = open(R + "/driver.log", "a")

    def log(s):
        logf.write(time.strftime("%F %T ") + s + "\n"); logf.flush()

    sp = R + "/state.json"
    st = json.load(open(sp)) if os.path.exists(sp) else {
        "turn": 0, "passes_used": 0, "python_runs": 0, "prev_id": None, "conv_turns": 0, "handoffs": 0,
        "done": False, "budget": a.passes, "model": a.model, "effort": a.effort}
    if st["done"]:
        print("run already finished"); return
    save = lambda: json.dump(st, open(sp, "w"), indent=1)
    dom.setup(R)
    kern = Kernel(dom.KERNEL_PYTHON, work, dom.PRELUDE)
    task = dom.TASK.format(passes=a.passes, max_python=a.max_python, target=getattr(dom, "TARGET", ""))
    open(R + "/message.txt", "w").write(task)

    def handoff_msg(reason):
        notes = open(work + "/notes.md").read()[-20000:] if os.path.exists(work + "/notes.md") else "(none)"
        return (f"[{reason}: this is a new conversation. Your kernel variables are gone but /work is intact.]\n\n"
                + task + "\n\n" + EXPLAIN + f"\n\nSTATUS: {st['passes_used']} of {st['budget']} forward passes and "
                f"{st['python_runs']} of {a.max_python} code runs used.\n\nYOUR NOTES (/work/notes.md):\n" + notes)

    if st["turn"] == 0:
        msg = task + "\n\n" + EXPLAIN
    else:  # resumed after a crash: start a fresh conversation
        st["prev_id"] = None; st["conv_turns"] = 0; st["handoffs"] += 1
        msg = handoff_msg("The harness was restarted")
    final = False
    while st["turn"] < a.max_turns:
        st["turn"] += 1; T = f"{R}/turns/t{st['turn']:03d}"; os.makedirs(T, exist_ok=True)
        open(T + "/prompt.txt", "w").write(msg)
        t0 = time.time()
        rid, text, summ, usage = call_llm(a.model, a.effort, msg, st["prev_id"], log)
        st["prev_id"] = rid; st["conv_turns"] += 1
        open(T + "/answer.md", "w").write(text)
        if summ:
            open(T + "/reasoning_summary.md", "w").write(summ)
        json.dump({"response_id": rid, "usage": usage, "wall_s": round(time.time() - t0, 1)}, open(T + "/run.json", "w"))
        log(f"turn {st['turn']} answered ({len(text)} chars)")
        m = BLOCK.search(text)
        if final:
            # after the budget: run final cleanup code (no forward passes), then ask again for the code-free answer
            if m is None or m.group(1) != "python" or st.get("final_code_runs", 0) >= 3:
                st["done"] = True; save(); log("finished"); break
            st["final_code_runs"] = st.get("final_code_runs", 0) + 1
            out, secs = kern.run(m.group(2))
            with open(R + "/tool_calls.jsonl", "a") as f:
                f.write(json.dumps({"turn": st["turn"], "t": time.time(), "kind": "python", "final": True,
                                    "code": m.group(2), "out": out[:20000], "s": secs}) + "\n")
            msg = (out + "\n\n[final code run executed. Now reply WITHOUT a code block: what is submitted in "
                   "/work/edited, what you changed, the evidence, and why you expect it to help.]")
            save(); continue
        rec = {"turn": st["turn"], "t": time.time()}
        if m is None:
            if st["passes_used"] >= st["budget"]:
                st["done"] = True; save(); log("finished (final answer)"); break
            out = ("No code or forward block found. Reply with exactly one ```python or ```forward block. "
                   f"{st['budget'] - st['passes_used']} forward passes remain; the session ends only when all are used.")
        elif m.group(1) == "python":
            if st["python_runs"] >= a.max_python:
                out = "[code budget exhausted: no more python runs]"
            else:
                st["python_runs"] += 1
                out, secs = kern.run(m.group(2))
                rec.update(kind="python", code=m.group(2), out=out[:20000], s=secs)
        else:
            try:
                req = json.loads(m.group(2))
            except Exception as e:
                req = None; out = f"[forward request is not valid JSON: {e}; it did not count]"
            if req is not None:
                if st["passes_used"] >= st["budget"]:
                    out = "[forward budget exhausted]"
                else:
                    k = st["passes_used"] + 1
                    try:
                        out = dom.forward(req, R, k)
                        st["passes_used"] = k
                    except ValueError as e:
                        out = f"[forward request refused: {e}; it did not count]"
                    except Exception:
                        out = "[forward tool failed (harness error); it did not count]\n" + traceback.format_exc()[-3000:]
                    rec.update(kind="forward", request=req, out=out[:20000], k=k)
        with open(R + "/tool_calls.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        status = f"\n\n[status: {st['passes_used']}/{st['budget']} forward passes, {st['python_runs']}/{a.max_python} code runs used]"
        if st["passes_used"] >= st["budget"] and not final:
            final = True
            status += ("\n\nAll forward passes are used. Make your final choice now: the tensors in /work/edited are "
                       "submitted. If /work/edited is not yet what you want to submit (e.g. you decided to revert), "
                       "reply with one ```python block that fixes it (up to 3 such blocks; no more forward passes); "
                       "otherwise reply WITHOUT a code block, saying what you changed, the evidence, and why you "
                       "expect it to help.")
        msg = out + status
        if st["conv_turns"] >= a.handoff_every and not final:
            st["prev_id"] = None; st["conv_turns"] = 0; st["handoffs"] += 1
            msg = "OUTPUT OF YOUR LAST BLOCK:\n" + msg + "\n\n" + handoff_msg("Conversation limit reached")
        save()
    kern.kill()


if __name__ == "__main__":
    main()
