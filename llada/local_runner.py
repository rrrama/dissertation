#!/usr/bin/env python3
"""Local GPU job supervisor — the replacement for slurm (SLURM_MIGRATION.md §1).

``batch_train.py`` no longer submits to a queue. It builds a list of `Job`s and
hands them here, and this module blocks until they are done, dispatching each
into free GPUs as they come available. Run it inside tmux::

    tmux new -s tier0
    cd llada && python batch_train.py --mode all --config configs/tier0_lr.yaml

Design notes, because the shape of this is load-bearing:

* **No durable state of its own.** The only record of progress is the output tree
  that already existed under slurm. Recovery from a dead supervisor is re-running
  the identical command: `_is_trained` skips finished runs, and
  `resume_from_checkpoint` picks up partly-trained ones. There is no queue file to
  corrupt and no stale PID to reap.

* **A run is one queue entry with two phases**, not two independent jobs. The
  train phase runs, and only on exit 0 does the benchmark phase run, in the same
  GPU slot. That is precisely what ``--dependency=afterok:<id>
  --kill-on-invalid-dep=yes`` bought on slurm, minus the scheduler. It also means
  the benchmark inherits a warm page cache from its own training job.

* **Children get their own session** (`start_new_session=True`), so a Ctrl-C at
  the supervisor does not tear down an 8-hour training run by accident. The
  handler is explicit instead: first Ctrl-C drains, second one kills.
"""

import os
import shutil
import signal
import socket
import subprocess
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import paths

POLL_SECONDS = 1.0

# Four concurrent jobs each loading an 8B model in bf16 spike CPU RAM together
# (~16 GB per rank, transient, `low_cpu_mem_usage=True`). Staggering dispatch
# makes the peaks miss each other; against an 8 h run it costs nothing.
DEFAULT_STAGGER_SECONDS = 60

# Minimum free space on the outputs volume before dispatching. Adapters are tiny;
# optimizer-state checkpoints are not, and several runs write them concurrently.
MIN_FREE_GB = 50


# --------------------------------------------------------------------------- #
# Job model                                                                   #
# --------------------------------------------------------------------------- #


@dataclass
class Phase:
    """One `train.py` invocation: a mode and the generated script that runs it."""

    mode: str
    script: str


@dataclass
class Job:
    """One run: its phases, executed in order, all in the same GPU slot."""

    name: str
    run_dir: str
    phases: List[Phase]
    n_gpus: int = 1

    # Mutable dispatch state, owned by `run_queue`.
    devices: List[str] = field(default_factory=list)
    proc: Optional[subprocess.Popen] = None
    phase_index: int = 0
    started_at: float = 0.0
    status: str = "queued"  # queued | running | done | failed | cancelled
    log_file: Optional[object] = None

    @property
    def phase(self) -> Optional[Phase]:
        if self.phase_index < len(self.phases):
            return self.phases[self.phase_index]
        return None


# --------------------------------------------------------------------------- #
# GPU pool                                                                    #
# --------------------------------------------------------------------------- #


def visible_devices() -> List[str]:
    """The GPU ids this supervisor may hand out.

    Honours an inherited ``CUDA_VISIBLE_DEVICES`` so that restricting the
    supervisor restricts its children: a child's value is interpreted against
    *physical* devices, so the ids from the parent's list are what must be passed
    down, not 0..n-1.
    """
    env = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if env:
        return [d.strip() for d in env.split(",") if d.strip()]

    if shutil.which("nvidia-smi") is None:
        return []
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--list-gpus"], stderr=subprocess.DEVNULL, text=True
        )
    except subprocess.CalledProcessError:
        return []
    return [str(i) for i in range(len(out.strip().splitlines()))]


class GpuPool:
    """Hands out GPU ids. First-fit, and a job holds its slot across all phases."""

    def __init__(self, devices: Sequence[str]):
        self.all = list(devices)
        self.free = list(devices)

    def acquire(self, n: int) -> Optional[List[str]]:
        if n > len(self.free):
            return None
        taken, self.free = self.free[:n], self.free[n:]
        return taken

    def release(self, devices: Sequence[str]) -> None:
        self.free.extend(devices)
        # Keep the free list in the pool's original order so run→GPU assignment is
        # reproducible across supervisor restarts rather than dependent on the
        # order jobs happened to finish in.
        self.free.sort(key=self.all.index)

    @property
    def in_use(self) -> int:
        return len(self.all) - len(self.free)


def _free_port() -> int:
    """A port nothing is listening on, for this job's c10d rendezvous.

    ``torchrun --standalone`` picks its own endpoint, and which one depends on the
    torch version -- older releases pin 29400, which two concurrent jobs on one box
    then collide over, intermittently, partway into a sweep. Under slurm this could
    not happen (one job per node). The generated scripts therefore ask for c10d
    explicitly on a port chosen here.

    The gap between probing and the child binding is a race in principle. At this
    concurrency it does not lose, and if it ever does, torchrun fails immediately
    and loudly rather than corrupting anything.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# --------------------------------------------------------------------------- #
# Preflight                                                                   #
# --------------------------------------------------------------------------- #


class PreflightError(RuntimeError):
    pass


def preflight(jobs: Sequence[Job], devices: Sequence[str]) -> None:
    """Fail before dispatching anything, rather than N jobs failing 30 s in.

    Six jobs each dying immediately, four hours after you detached from tmux, is
    the outcome this is buying against.
    """
    problems = []

    if not devices:
        problems.append(
            "no GPUs visible (nvidia-smi found none, and CUDA_VISIBLE_DEVICES is unset)"
        )

    needed = max((j.n_gpus for j in jobs), default=0)
    if devices and needed > len(devices):
        offenders = sorted({j.name for j in jobs if j.n_gpus > len(devices)})
        problems.append(
            f"{len(offenders)} run(s) ask for {needed} GPUs but only {len(devices)} "
            f"are visible: {', '.join(offenders)}.\n"
            f"    Lower `nproc_per_node` in the config. A job that can never be "
            f"scheduled would otherwise wait forever."
        )

    activate = os.path.join(paths.VENV_DIR, "bin", "activate")
    if not os.path.exists(activate):
        problems.append(
            f"no venv at {paths.VENV_DIR}. Run: bash scripts/bootstrap.sh"
        )
    elif not os.path.exists(paths.BOOTSTRAP_MARKER):
        problems.append(
            f"{paths.BOOTSTRAP_MARKER} is missing, so this instance was never "
            f"bootstrapped (or SCRATCH_ROOT moved). Run: bash scripts/bootstrap.sh"
        )
    else:
        problems.extend(_check_peft())

    free_gb = _free_gb(paths.OUTPUTS_ROOT)
    if free_gb is not None and free_gb < MIN_FREE_GB:
        problems.append(
            f"only {free_gb} GB free on {paths.OUTPUTS_ROOT}; checkpoints from "
            f"concurrent runs need ~{MIN_FREE_GB} GB"
        )

    if problems:
        raise PreflightError(
            "preflight failed:\n" + "\n".join(f"  - {p}" for p in problems)
        )


def _check_peft() -> List[str]:
    """Assert the vendored fork imports, and imports from *this* checkout.

    In a subprocess so that `batch_train.py` never imports torch itself. A stray
    PyPI peft imports perfectly well and then fails inside `_build_peft_config`,
    which is a much worse place to find out.
    """
    code = (
        "import peft;"
        "from peft import LoraConfig, LorTaConfig, NALorTaConfig, NARAConfig;"
        "print(peft.__file__)"
    )
    try:
        out = subprocess.run(
            [os.path.join(paths.VENV_DIR, "bin", "python"), "-c", code],
            capture_output=True,
            text=True,
            timeout=180,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [f"could not check the peft import: {exc}"]

    if out.returncode != 0:
        tail = (out.stderr or "").strip().splitlines()
        return [
            "the vendored peft fork does not import: "
            + (tail[-1] if tail else "unknown error")
            + "\n    Re-run: bash scripts/bootstrap.sh"
        ]

    peft_file = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
    if not peft_file.startswith(paths.PEFT_DIR + os.sep):
        return [
            f"peft resolves to {peft_file}, outside {paths.PEFT_DIR}. The tuners "
            f"being trained would not be the ones in this checkout."
        ]
    return []


def _free_gb(path: str) -> Optional[int]:
    try:
        usage = shutil.disk_usage(path)
    except OSError:
        return None
    return int(usage.free / 1e9)


# --------------------------------------------------------------------------- #
# The dispatch loop                                                           #
# --------------------------------------------------------------------------- #


def _stamp() -> str:
    return time.strftime("%H:%M:%S")


def _say(message: str) -> None:
    print(f"[{_stamp()}] {message}", flush=True)


def _elapsed(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


def _child_env(job: Job, threads_per_rank: int) -> dict:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(job.devices)
    env["LORTA_RDZV_PORT"] = str(_free_port())
    env["LORTA_RUN_NAME"] = job.name
    # Default OMP threading is per-process core count, which thrashes badly at
    # 3-4x oversubscription on one box. Under slurm, --cpus-per-task handled this.
    env["OMP_NUM_THREADS"] = str(threads_per_rank)
    return env


def _launch(job: Job, threads_per_rank: int) -> None:
    phase = job.phase
    log_path = os.path.join(job.run_dir, "logs", f"{phase.mode}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Append rather than truncate: a resumed run's second attempt belongs in the
    # same file as its first, and the header says where one ends.
    log = open(log_path, "a", buffering=1)
    log.write(
        f"\n{'=' * 78}\n"
        f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {job.name}  phase={phase.mode}  "
        f"gpus={','.join(job.devices)}\n{'=' * 78}\n"
    )
    log.flush()

    job.proc = subprocess.Popen(
        ["bash", phase.script],
        cwd=paths.LLADA_DIR,
        env=_child_env(job, threads_per_rank),
        stdout=log,
        stderr=subprocess.STDOUT,
        # Its own session, so Ctrl-C here does not signal a training run.
        start_new_session=True,
    )
    job.log_file = log  # closed in _reap
    job.started_at = time.time()
    job.status = "running"
    _say(
        f"start   {job.name} [{phase.mode}] gpu {','.join(job.devices)} "
        f"pid {job.proc.pid} -> {paths.short(log_path)}"
    )


def _reap(job: Job, pool: GpuPool, threads_per_rank: int) -> None:
    """Handle a finished phase: advance to the next one, or free the slot."""
    code = job.proc.returncode
    took = _elapsed(time.time() - job.started_at)
    mode = job.phases[job.phase_index].mode
    if job.log_file is not None:
        job.log_file.close()
        job.log_file = None
    job.proc = None

    if code != 0:
        # The `afterok` half of the old dependency: a failed train phase means the
        # benchmark never runs, rather than scoring an adapter that was not saved.
        job.status = "failed"
        pool.release(job.devices)
        skipped = [p.mode for p in job.phases[job.phase_index + 1 :]]
        note = f"; skipping {', '.join(skipped)}" if skipped else ""
        _say(f"FAILED  {job.name} [{mode}] exit {code} after {took}{note}")
        job.devices = []
        return

    job.phase_index += 1
    if job.phase is None:
        job.status = "done"
        pool.release(job.devices)
        _say(f"done    {job.name} [{mode}] in {took}")
        job.devices = []
        return

    _say(f"ok      {job.name} [{mode}] in {took}; chaining {job.phase.mode}")
    _launch(job, threads_per_rank)


def run_queue(
    jobs: Sequence[Job],
    devices: Optional[Sequence[str]] = None,
    max_concurrent: Optional[int] = None,
    stagger_seconds: float = DEFAULT_STAGGER_SECONDS,
    dry_run: bool = False,
) -> List[Tuple[str, str]]:
    """Run every job to completion. Blocks. Returns ``[(name, status), ...]``.

    Dispatch is first-fit over the queue *in submission order*: the first job
    whose GPU request fits the free pool goes next. Scanning past a job that does
    not fit avoids idling three GPUs behind one 4-GPU run; for a homogeneous sweep
    it is plain FIFO.
    """
    jobs = list(jobs)
    if not jobs:
        print("Nothing to run.")
        return []

    devices = list(devices if devices is not None else visible_devices())

    if dry_run:
        _print_plan(jobs, devices)
        return [(j.name, "dry-run") for j in jobs]

    preflight(jobs, devices)

    pool = GpuPool(devices)
    threads_per_rank = max(1, (os.cpu_count() or len(devices)) // max(1, len(devices)))
    limit = max_concurrent or len(devices)

    _say(
        f"{len(jobs)} job(s) over {len(devices)} GPU(s) "
        f"[{', '.join(devices)}], {threads_per_rank} thread(s)/rank"
    )

    draining = {"flag": False}

    def _on_sigint(signum, frame):
        if draining["flag"]:
            _say("second interrupt: terminating running jobs")
            for job in jobs:
                if job.proc is not None:
                    _terminate(job)
            raise KeyboardInterrupt
        draining["flag"] = True
        running = [j.name for j in jobs if j.status == "running"]
        _say(
            "interrupt: no new jobs will start. Still running: "
            + (", ".join(running) if running else "nothing")
            + "\n           Interrupt again to kill them; they survive otherwise."
        )

    previous = signal.signal(signal.SIGINT, _on_sigint)
    last_launch = 0.0
    try:
        while True:
            for job in jobs:
                if job.proc is not None and job.proc.poll() is not None:
                    _reap(job, pool, threads_per_rank)

            running = sum(1 for j in jobs if j.status == "running")
            if not draining["flag"] and running < limit:
                waited = time.time() - last_launch
                if waited >= stagger_seconds or running == 0:
                    for job in jobs:
                        if job.status != "queued":
                            continue
                        got = pool.acquire(job.n_gpus)
                        if got is None:
                            continue
                        job.devices = got
                        _launch(job, threads_per_rank)
                        last_launch = time.time()
                        break

            if all(j.status in ("done", "failed", "cancelled") for j in jobs):
                break
            if draining["flag"] and not any(j.status == "running" for j in jobs):
                for job in jobs:
                    if job.status == "queued":
                        job.status = "cancelled"
                break
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        for job in jobs:
            if job.status in ("queued", "running"):
                job.status = "cancelled"
    finally:
        signal.signal(signal.SIGINT, previous)

    _print_result(jobs)
    return [(j.name, j.status) for j in jobs]


def _terminate(job: Job) -> None:
    """SIGTERM the child's whole process group — torchrun plus every rank."""
    if job.proc is None:
        return
    try:
        os.killpg(os.getpgid(job.proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass


def _print_plan(jobs: Sequence[Job], devices: Sequence[str]) -> None:
    print(f"\n[dry-run] {len(jobs)} job(s); {len(devices)} GPU(s) visible "
          f"[{', '.join(devices) or 'none'}]")
    for job in jobs:
        chain = " && ".join(p.mode for p in job.phases) or "(nothing)"
        print(f"  {job.name:<28} {job.n_gpus} gpu  {chain}")
        for phase in job.phases:
            print(f"      {paths.short(phase.script)}")
    print("\n[dry-run] nothing was launched.")


def _print_result(jobs: Sequence[Job]) -> None:
    by_status = {}
    for job in jobs:
        by_status.setdefault(job.status, []).append(job.name)
    print()
    _say("queue finished: " + ", ".join(
        f"{len(names)} {status}" for status, names in sorted(by_status.items())
    ))
    for status in ("failed", "cancelled"):
        for name in by_status.get(status, []):
            print(f"  {status}: {name}")
    if "failed" in by_status:
        print(
            "\nLogs are in each run's logs/ directory. Re-running the same "
            "batch_train.py command retries only the unfinished runs."
        )


def exit_code(results: Sequence[Tuple[str, str]]) -> int:
    """1 if anything failed, so a wrapper script can tell."""
    return 1 if any(status == "failed" for _, status in results) else 0
