#!/usr/bin/env python3
"""Materialize the archived experiment layout on a new machine, without launching it."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "experiments"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())

    def path(value):
        p = Path(value).expanduser()
        return (args.config.resolve().parent / p).resolve() if not p.is_absolute() else p.resolve()

    workspace = path(config.get("workspace", "runtime"))
    if workspace.exists():
        raise FileExistsError(f"Refusing to overwrite workspace: {workspace}")
    work = path(config.get("fastwam_work", str(workspace / "vendor/fastwam-work")))
    study, pi05 = workspace / "study", workspace / "pi05"
    recovery, pro = workspace / "pi05_recovery", workspace / "pi05_pro"
    replacements = {
        "/mnt/SSD2_16TB/jusheng/Documents/Codex/2026-10-02/vla-wam-cvpr-gemini-api-key/work": str(work),
        "/mnt/SSD_16TB/jusheng/wsp_recover": str(recovery),
        "/mnt/SSD_16TB/jusheng/wsp_pi05": str(pi05),
        "/mnt/SSD_16TB/jusheng/wsp_pro": str(pro),
        "/mnt/SSD2_16TB/jusheng/wsp": str(study),
    }
    manifest = []

    def copy(source, destination):
        text = source.read_text()
        for original, relocated in replacements.items():
            text = text.replace(original, relocated)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text)
        manifest.append({"source": str(source.relative_to(REPO)),
                         "output": str(destination.relative_to(workspace)),
                         "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                         "output_sha256": hashlib.sha256(destination.read_bytes()).hexdigest()})

    for backend, target in [("fastwam", study / "fastwam"), ("pi05", pi05),
                            ("pi05_recovery", recovery), ("pi05_libero_pro", pro)]:
        for source in (SOURCE / "backends" / backend).iterdir():
            if source.suffix == ".py" or source.name == "README.md":
                copy(source, target / source.name)
    for target in [study / "harness", pi05 / "harness"]:
        for source in (SOURCE / "harness").glob("*.py"):
            copy(source, target / source.name)
        for source in (SOURCE / "domains/fastwam").glob("*.py"):
            copy(source, target / source.name)
    for source in (SOURCE / "domains/pi05").glob("*.py"):
        copy(source, pi05 / "harness" / source.name)
    practice = REPO / "bench/practice_variants.json"
    for target in [study / "bench", pi05 / "bench"]:
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(practice, target / practice.name)

    for name, target in [("kernel_python", study / "envs/kernel/bin/python"),
                         ("fastwam_python", study / "envs/fastwam_plus/bin/python"),
                         ("pi05_python", pi05 / "venv/bin/python")]:
        value = config.get(name) or sys.executable
        executable = shutil.which(value) or str(path(value))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(Path(executable).resolve())
    env = {"WSP_PI05": str(pi05), "WSP_STUDY_ROOT": str(study), "MUJOCO_GL": "egl"}
    (workspace / "env.sh").write_text("\n".join(f"export {k}={shlex.quote(v)}" for k, v in env.items()) + "\n")
    (workspace / "configured-files.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(manifest)} source files in {workspace}")
    print("No dependencies installed, checkpoints downloaded, policy servers started, or LLM requests sent.")
    print(f"Next: source {shlex.quote(str(workspace / 'env.sh'))}")


if __name__ == "__main__":
    main()
