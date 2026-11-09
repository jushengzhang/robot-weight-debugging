"""Persistent Python kernel for the weight editor (notebook-like: variables survive between blocks).

The parent talks to a child process over pipes with length-prefixed JSON. The child runs with a cleared environment
(no proxy variables, so no internet from 120; CUDA hidden), an address-space limit, and a per-run wall-clock limit;
on timeout or crash the child is restarted and the caller is told that variables were lost.
"""
import json, os, resource, struct, subprocess, sys, time

CHILD = r'''
import ast, contextlib, io, json, struct, sys, traceback
_in, _out = sys.stdin.buffer, sys.stdout.buffer
sys.stdout = io.TextIOWrapper(io.BytesIO())  # nothing may write to the protocol pipe directly
G = {"__name__": "__main__"}
def _recv():
    h = _in.read(4)
    if len(h) < 4: sys.exit(0)
    return json.loads(_in.read(struct.unpack(">I", h)[0]))
def _send(o):
    b = json.dumps(o).encode(); _out.write(struct.pack(">I", len(b)) + b); _out.flush()
while True:
    msg = _recv()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            tree = ast.parse(msg["code"])
            last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
            exec(compile(tree, "<cell>", "exec"), G)
            if last is not None:
                v = eval(compile(ast.Expression(last.value), "<cell>", "eval"), G)
                if v is not None: print(repr(v))
        except BaseException:
            traceback.print_exc()
    _send({"out": buf.getvalue()})
'''


class Kernel:
    def __init__(self, python, cwd, prelude="", timeout=900, mem_gb=96, max_out=10_000, env_extra=None):
        self.python, self.cwd, self.prelude = python, cwd, prelude
        self.timeout, self.mem, self.max_out = timeout, mem_gb << 30, max_out
        self.env = {"PATH": "/usr/bin:/bin", "HOME": cwd, "OMP_NUM_THREADS": "16", "MKL_NUM_THREADS": "16",
                    "CUDA_VISIBLE_DEVICES": "", "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1", **(env_extra or {})}
        self.p = None
        self.start()

    def _limits(self):
        resource.setrlimit(resource.RLIMIT_AS, (self.mem, self.mem))
        os.setsid()

    def start(self):
        self.p = subprocess.Popen([self.python, "-I", "-c", CHILD], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, cwd=self.cwd, env=self.env, preexec_fn=self._limits)
        if self.prelude:
            out = self._call(self.prelude, 600)
            if out.strip():
                raise RuntimeError("prelude failed:\n" + out)

    def _call(self, code, timeout):
        import select
        b = json.dumps({"code": code}).encode()
        self.p.stdin.write(struct.pack(">I", len(b)) + b); self.p.stdin.flush()
        r, _, _ = select.select([self.p.stdout], [], [], timeout)
        if not r:
            raise TimeoutError
        h = self.p.stdout.read(4)
        if len(h) < 4:
            raise EOFError
        return json.loads(self.p.stdout.read(struct.unpack(">I", h)[0]))["out"]

    def kill(self):
        if self.p and self.p.poll() is None:
            try:
                os.killpg(self.p.pid, 9)
            except ProcessLookupError:
                pass
        self.p = None

    def run(self, code):
        t0 = time.time()
        try:
            out = self._call(code, self.timeout)
        except TimeoutError:
            self.kill(); self.start()
            out = f"[timed out after {self.timeout} s; the kernel was restarted and its variables were lost]"
        except (EOFError, BrokenPipeError, OSError):
            self.kill(); self.start()
            out = "[the kernel died (probably the memory limit); it was restarted and its variables were lost]"
        out = out.encode("ascii", "replace").decode("ascii")
        if len(out) > self.max_out:
            out = out[:self.max_out] + f"\n[output cut at {self.max_out:,} characters]"
        return (out or "(no output)"), round(time.time() - t0, 2)
