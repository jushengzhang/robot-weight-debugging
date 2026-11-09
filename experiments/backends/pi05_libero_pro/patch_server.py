#!/usr/bin/env python
"""Build pro_server.py = COPY of wsp_pi05/policy_server.py + LIBERO-PRO episodes (ep has "ptype").
   python patch_server.py <src policy_server.py> <dst pro_server.py>   (every replacement must hit exactly once)"""
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)


rep('"""pi0.5 (openpi `pi05_libero`, PyTorch port) policy server for LIBERO-Plus:',
    '"""[wsp_pro COPY of wsp_pi05/policy_server.py + LIBERO-PRO episodes (ep has "ptype"); see README.md]\n'
    'pi0.5 (openpi `pi05_libero`, PyTorch port) policy server for LIBERO-Plus:')
rep('WSP = Path(os.environ.get("WSP_PI05", str(Path.home() / "wsp_pi05")))\n',
    'WSP = Path(os.environ.get("WSP_PI05", str(Path.home() / "wsp_pi05")))\n'
    'WSP_PRO = Path(__file__).resolve().parent\n')
rep('        if os.environ.get("WSP_ORIG"):  # original LIBERO: no LIBERO-Plus classification',
    '        if os.environ.get("WSP_ORIG") or os.environ.get("WSP_PRO_MODE"):  # LIBERO-orig / LIBERO-PRO: no LIBERO-Plus classification')
rep('        self.device = "cuda"\n',
    '        self.device = os.environ.get("WSP_DEVICE", "cuda")  # wsp_pro: "cpu" fallback when GPUs 6/7 are full\n')
rep('''            "gpu_mem_allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "gpu_mem_reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "gpu_max_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,''',
    '''            "gpu_mem_allocated_gb": (torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0),
            "gpu_mem_reserved_gb": (torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0.0),
            "gpu_max_reserved_gb": (torch.cuda.max_memory_reserved() / 1e9 if torch.cuda.is_available() else 0.0),
            "device": self.device, "libero": __import__("libero.libero").libero.__file__,''')
rep('''                "gpu_max_reserved_gb": torch.cuda.max_memory_reserved() / 1e9}''',
    '''                "gpu_max_reserved_gb": (torch.cuda.max_memory_reserved() / 1e9 if torch.cuda.is_available() else 0.0)}''')
rep('''    def run_episode(self, ep, *, recorder, out_dir, save_video, save_frames, tag, language):
''', '''    def run_episode(self, ep, *, recorder, out_dir, save_video, save_frames, tag, language):
        if "ptype" in ep:  # wsp_pro: LIBERO-PRO episode
            return self.run_pro_episode(ep, recorder=recorder, out_dir=out_dir, save_video=save_video,
                                        save_frames=save_frames, tag=tag, language=language)
''')

PRO = r'''
    # ---------------- LIBERO-PRO episodes (wsp_pro) ----------------
    # Episode = {"suite": "libero_goal", "ptype": object|position|semantic|task|environment|original,
    #            "task": <task name> (or "task_idx"), "init_idx": 0..49, "seed": 7[, "max_steps", "language"]}
    # Protocol = run_episode above (openpi examples/libero/main.py; = LIBERO-PRO's OpenVLA-OFT runner):
    # env from the LIBERO-PRO suite '<suite>_<lan|object|swap|task|env>' BDDL, env.seed(seed), reset,
    # set_init_state(init_states[init_idx]) from the matching LIBERO-PRO init file, 10 settle steps,
    # chunk 10 / replan 5, max steps 220/280/300/520 (LIBERO-PRO TASK_MAX_STEPS), success = env done.
    # language: "pro" (default; also for "strip"/"clean") = the perturbed BDDL (:language) (lowercased) when LIBERO-PRO
    #   changed it (semantic/task), else the standard LIBERO instruction task.language;
    #   "filename" = task.language always (literal upstream run_libero_eval); "raw" = BDDL (:language) verbatim.
    PRO_SUFFIX = {"object": "object", "position": "swap", "semantic": "lan", "task": "task", "environment": "env"}

    @staticmethod
    def _free_obj_pos(inner):
        """world xyz of every free-joint object, read from qpos (body_xpos is stale right after set_init_state;
        reading qpos needs no sim.forward(), so the dynamics are untouched)"""
        sim, out = inner.sim, {}
        m = sim.model
        for n, b in getattr(inner, "obj_body_id", {}).items():
            ja = int(m.body_jntadr[b])
            if ja < 0 or int(m.jnt_type[ja]) != 0:
                continue
            qa = int(m.jnt_qposadr[ja])
            out[n] = np.array(sim.data.qpos[qa:qa + 3], dtype=np.float64)
        return out

    @staticmethod
    def _bddl_lang(text):
        m = re.search(r"\(:language\s+(.+?)\)", text, flags=re.S)
        return " ".join(m.group(1).split()) if m else None

    def run_pro_episode(self, ep, *, recorder, out_dir, save_video, save_frames, tag, language):
        from libero.libero.envs.bddl_utils import robosuite_parse_problem
        suite_name = ep["suite"]
        ptype = ep.get("ptype", "original")
        pro_suite = suite_name if ptype == "original" else f"{suite_name}_{self.PRO_SUFFIX[ptype]}"
        seed = int(ep.get("seed", 7))
        init_idx = int(ep.get("init_idx", 0))
        suite = self.variants.suite(pro_suite)
        names = [t.name for t in suite.tasks]
        task_idx = names.index(ep["task"]) if "task" in ep else int(ep["task_idx"])
        task = suite.get_task(task_idx)
        key = ep.get("key") or f"{suite_name}/{ptype}/{task.name}/{init_idx:02d}"
        res = {"key": key, "suite": suite_name, "ptype": ptype, "category": ptype, "pro_suite": pro_suite,
               "task": task.name, "task_idx": task_idx, "variant_id": key, "seed": seed, "init_idx": init_idx,
               "arm": tag, "name": task.name, "success": False, "steps": 0, "failure": None}
        stem = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{suite_name}_{ptype}_{task_idx}_i{init_idx}_s{seed}_{tag}")
        t0 = time.time()
        env = None
        try:
            self._seed_all(seed)
            gen = torch.Generator(device="cpu").manual_seed(seed)  # flow-matching noise, fresh draw per query
            bddl_dir = Path(get_libero_path("bddl_files"))
            bddl = bddl_dir / task.problem_folder / task.bddl_file
            text = bddl.read_text()
            res["bddl_sha256"] = hashlib.sha256(text.encode()).hexdigest()[:16]
            # upstream Benchmark.get_task_init_states uses torch.load(path) (weights_only error on torch>=2.6)
            init_states = torch.load(Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file,
                                     weights_only=False)
            for attempt in range(5):
                try:
                    env, _ = self.make_env(task, seed)
                    break
                except Exception as exc:
                    if exc.__class__.__name__ != "RandomizationError" or attempt == 4:
                        raise
            pl = self._bddl_lang(text) or ""
            orig_text = (bddl_dir / suite_name / task.bddl_file).read_text()
            ol = self._bddl_lang(orig_text) or ""
            changed = pl.lower().rstrip(".") != ol.lower().rstrip(".")
            mode = ep.get("language", language)
            if mode == "raw":
                instr = pl
            elif mode == "filename":
                instr = task.language
            else:
                mode = "pro"
                instr = pl.lower().rstrip(".") if changed else task.language
            res.update(instruction=instr, instruction_bddl=pl, instruction_orig=task.language,
                       instruction_changed=changed, language_mode=mode)
            # task perturbation: also track the ORIGINAL task's goal (memorization indicator)
            orig_goal = robosuite_parse_problem(str(bddl_dir / suite_name / task.bddl_file))["goal_state"] \
                if ptype == "task" else None
            inner = env.env

            def orig_goal_met():
                try:
                    return all(bool(inner._eval_predicate(g)) for g in orig_goal)
                except Exception:
                    return None

            max_steps = int(ep.get("max_steps", MAX_STEPS[suite_name]))
            frame_every = int(save_frames) if save_frames else 0
            fdir = None
            if frame_every and out_dir:
                fdir = Path(out_dir) / f"{stem}_frames"
                fdir.mkdir(parents=True, exist_ok=True)
            env.reset()
            obs = env.set_init_state(init_states[init_idx % len(init_states)])
            pos0 = self._free_obj_pos(inner)
            res["n_free_objects"] = len(pos0)
            if recorder is not None:
                recorder.reset()
            frames, pending, done, t, n_q, trace, saved = [], [], False, 0, 0, [], []
            eef0, path_len, orig_hit = None, 0.0, None
            while t < max_steps + NUM_STEPS_WAIT:
                if t < NUM_STEPS_WAIT:
                    obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    if t == NUM_STEPS_WAIT:  # objects that drift during settling (unstable perturbed scene)
                        cur = self._free_obj_pos(inner)  # xy only: LIBERO init states spawn objects ~7 cm high (they drop)
                        res["settle_drift_m"] = round(max([float(np.linalg.norm((cur[n] - pos0[n])[:2])) for n in pos0] or [0.0]), 4)
                        pos0 = cur  # obj_moved_m below = displacement from the settled pose
                        res["done_at_t0"] = bool(done)
                        eef0 = np.array(obs["robot0_eef_pos"], dtype=np.float64)
                    continue
                st = t - NUM_STEPS_WAIT
                element = None
                if not pending or save_video:
                    element = self.make_element(obs, instr)
                if not pending:
                    if orig_goal is not None and orig_hit is None and orig_goal_met():
                        orig_hit = st
                    noise = torch.randn((self.action_horizon, self.action_dim), generator=gen, dtype=torch.float32).numpy()
                    chunk = self.infer(element, noise)
                    n_q += 1
                    pending = list(chunk[:REPLAN_STEPS])
                    if fdir is not None and st % frame_every == 0:
                        Image.fromarray(np.concatenate([element["observation/image"], element["observation/wrist_image"]], 1)).save(fdir / f"t{st:04d}.png")
                        torch.save({"element": element, "noise": noise, "seed": seed, "suite": suite_name,
                                    "variant_id": key, "ptype": ptype, "step": st, "arm": tag, "action_chunk": chunk,
                                    "replan_steps": REPLAN_STEPS}, fdir / f"t{st:04d}_inputs.pt")
                        saved.append(st)
                if save_video and st % 2 == 0:
                    frames.append(np.concatenate([element["observation/image"], element["observation/wrist_image"]], 1))
                a = pending.pop(0)
                trace.append(np.asarray(a, dtype=np.float64))
                prev = np.array(obs["robot0_eef_pos"], dtype=np.float64)
                obs, _, done, _ = env.step(np.asarray(a).tolist())
                path_len += float(np.linalg.norm(np.asarray(obs["robot0_eef_pos"]) - prev))
                if done:
                    break
                t += 1
            res["success"] = bool(done)
            res["steps"] = int(t - NUM_STEPS_WAIT + (1 if done else 0))
            res["n_queries"] = n_q
            res["termination"] = "success" if done else "max_steps"
            if not done:
                res["failure"] = "timeout"
            if orig_goal is not None:
                if orig_hit is None and orig_goal_met():
                    orig_hit = res["steps"]
                res["orig_goal_reached"] = orig_hit is not None
                res["orig_goal_step"] = orig_hit
            ah = np.asarray(trace, dtype=np.float32)
            res["action_sha1"] = hashlib.sha1(ah.tobytes()).hexdigest()[:16]
            if len(ah):  # cheap behaviour descriptors for failure-mode analysis
                g = ah[:, 6]
                res["grip_flips"] = int(np.sum(np.abs(np.diff(np.sign(g))) > 0))
                res["frac_grip_closed"] = round(float(np.mean(g > 0)), 3)
            res["eef_path_m"] = round(path_len, 4)
            if eef0 is not None:
                res["eef_net_m"] = round(float(np.linalg.norm(np.asarray(obs["robot0_eef_pos"]) - eef0)), 4)
            cur = self._free_obj_pos(inner)
            res["obj_moved_m"] = {n: round(float(np.linalg.norm(cur[n] - pos0[n])), 3) for n in pos0}  # 3D, from settled pose
            if out_dir is not None:
                np.save(Path(out_dir) / f"{stem}_actions.npy", ah)
                res["actions"] = str(Path(out_dir) / f"{stem}_actions.npy")
            if fdir is not None:
                res["frames_dir"] = str(fdir)
                res["frames_steps"] = saved
            if recorder is not None and out_dir is not None:
                p_ = Path(out_dir) / f"{stem}_acts.npz"
                res["activation_calls"] = recorder.dump(p_)
                res["activations"] = str(p_)
            if save_video and out_dir is not None and frames:
                import imageio
                vp = Path(out_dir) / f"{stem}_{'succ' if done else 'fail'}.mp4"
                imageio.mimwrite(str(vp), frames, fps=10)
                res["video"] = str(vp)
        except Exception as exc:
            res["failure"] = f"error: {exc!r}"
            res["termination"] = "error"
            res["traceback"] = traceback.format_exc()[-3000:]
        finally:
            if env is not None:
                with contextlib.suppress(Exception):
                    env.close()
        res["wall_s"] = round(time.time() - t0, 2)
        return res

    def _common(self, req):'''
rep('\n    def _common(self, req):', PRO)
open(dst, "w").write(s)
print("ok", len(s))
