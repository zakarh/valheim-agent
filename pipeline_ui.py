"""Tkinter control panel for the Valheim RLHF training/evaluation pipeline.

One tab per pipeline stage. Each tab builds a command line for the matching
script and runs it as a subprocess, streaming output into the shared console.
The console input box forwards lines to the running process's stdin, which is
how the interactive compare_rollouts.py labeling prompt is answered.

Only one process runs at a time: the live-control scripts all fight over the
focused Valheim window, so parallel runs would not make sense anyway.

Usage:
    python pipeline_ui.py
"""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import sys
import threading
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_GOALS = ["general", "explore", "gather_wood", "combat", "build"]

MODEL_FILES = {
    "behavior clone": Path("models/behavior_clone.pt"),
    "reward model": Path("models/reward_model.pt"),
    "rlhf dqn": Path("models/rlhf_dqn.zip"),
}


@dataclass
class Field:
    """One CLI option rendered as a form widget."""

    flag: str  # e.g. "--goal"; empty string marks a bool flag with no value
    label: str
    default: str = ""
    kind: str = "str"  # str | goal | bool | choice
    choices: tuple[str, ...] = ()
    help: str = ""
    var: tk.Variable = field(init=False, repr=False, default=None)


@dataclass
class Stage:
    title: str
    script: str
    blurb: str
    fields: list[Field]
    note: str = ""


def build_stages() -> list[Stage]:
    device = Field("--device", "Device", "auto", "choice", ("auto", "cpu", "cuda"))
    return [
        Stage(
            "1. Capture",
            "valheim_capture.py",
            "Confirm window capture works. Waits for valheim.exe, writes captures/latest.png, logs FPS.",
            [],
        ),
        Stage(
            "2. Record Human",
            "record_gameplay.py",
            "Record human demonstrations with a goal label for behavior cloning.",
            [
                Field("--goal", "Goal", "explore", "goal"),
                Field("--fps", "FPS", "10"),
                Field("--duration", "Duration (s, 0 = until stopped)", "600"),
                Field("--frame-width", "Frame width", "320"),
            ],
            note="Keep Valheim focused while recording.",
        ),
        Stage(
            "3. Train BC",
            "train_behavior_clone.py",
            "Train the goal-conditioned behavior-cloning model from recorded demonstrations.",
            [
                Field("--goals", "Goals (comma-separated, blank = all)", ""),
                Field("--epochs", "Epochs", "10"),
                Field("--batch-size", "Batch size", "64"),
                Field("--learning-rate", "Learning rate", "1e-4"),
                Field("--model-path", "Model path", "models/behavior_clone.pt"),
                device,
            ],
        ),
        Stage(
            "4. Run BC",
            "run_behavior_clone.py",
            "Drive Valheim live with the behavior-clone model.",
            [
                Field("--goal", "Goal", "explore", "goal"),
                Field("--model-path", "Model path", "models/behavior_clone.pt"),
                Field("--dry-run", "Dry run (no inputs sent)", "0", "bool"),
                Field("--device", "Device", "auto", "choice", ("auto", "cpu", "cuda")),
            ],
            note="Use a safe single-player world. Dry run prints predictions only.",
        ),
        Stage(
            "5. Record Rollout",
            "record_ai_rollout.py",
            "Record an AI episode (video + actions + metadata) under rollouts/<policy-name>/.",
            [
                Field("--policy", "Policy", "random", "choice", ("random", "behavior_clone", "dqn")),
                Field("--policy-name", "Policy name (rollout folder)", ""),
                Field("--goal", "Goal", "explore", "goal"),
                Field("--model-path", "Model path (bc .pt / dqn .zip)", "models/behavior_clone.pt"),
                Field("--duration", "Duration (s)", "60"),
                Field("--seed", "Seed", "7"),
                Field("--start-delay", "Start delay (s)", "5"),
                Field("--dry-run", "Dry run (no inputs sent)", "0", "bool"),
                Field("--device", "Device", "auto", "choice", ("auto", "cpu", "cuda")),
            ],
            note="Record two same-goal baselines with different seeds/names to compare.",
        ),
        Stage(
            "6. Compare A/B",
            "compare_rollouts.py",
            "Build side-by-side comparison videos and label which rollout is better.",
            [
                Field("--goal", "Goal filter", "explore", "goal"),
                Field("--policy-a", "Policy A filter", ""),
                Field("--policy-b", "Policy B filter", ""),
                Field("--pairs", "Pairs", "5"),
                Field("--max-seconds", "Max seconds per side", "60"),
                Field("--no-label", "Render videos only (no prompts)", "0", "bool"),
            ],
            note="Watch each video in comparisons/, then answer a/b/t/s/q in the console input below.",
        ),
        Stage(
            "7. Train Reward",
            "train_reward_model.py",
            "Train the goal-conditioned reward model from saved A/B preferences.",
            [
                Field("--goals", "Goals (comma-separated, blank = all)", ""),
                Field("--epochs", "Epochs", "20"),
                Field("--batch-size", "Batch size", "8"),
                Field("--learning-rate", "Learning rate", "1e-4"),
                Field("--model-path", "Model path", "models/reward_model.pt"),
                Field("--device", "Device", "auto", "choice", ("auto", "cpu", "cuda")),
            ],
        ),
        Stage(
            "8. Score Rollouts",
            "score_rollouts.py",
            "Sanity-check the reward model by ranking recorded rollouts. Output: preferences/rollout_scores.csv.",
            [
                Field("--goal", "Goal override (blank = each rollout's own)", "", "goal"),
                Field("--clips-per-rollout", "Clips per rollout", "8"),
                Field("--device", "Device", "auto", "choice", ("auto", "cpu", "cuda")),
            ],
        ),
        Stage(
            "9. Train RLHF",
            "train_rlhf_policy.py",
            "Train a DQN live against the learned reward. Valheim must be running and focused.",
            [
                Field("--goal", "Goal (single)", "explore", "goal"),
                Field("--goals", "Goals (comma-separated, overrides single)", ""),
                Field("--timesteps", "Timesteps", "10000"),
                Field("--reward-mode", "Reward mode", "delta", "choice", ("delta", "score")),
                Field("--reward-scale", "Reward scale", "1.0"),
                Field("--model-path", "Model path", "models/rlhf_dqn.zip"),
                Field("--no-input", "No input (smoke test)", "0", "bool"),
                Field("--device", "Device", "auto", "choice", ("auto", "cpu", "cuda")),
            ],
            note="After training, go back to stage 5 to record the new policy and repeat the loop.",
        ),
    ]


def scan_goals() -> list[str]:
    """Collect goal labels seen in dataset/rollout metadata plus the defaults."""
    goals = set(DEFAULT_GOALS)
    for meta_glob in ("datasets/*/meta.json", "rollouts/*/*/meta.json"):
        for meta_path in PROJECT_ROOT.glob(meta_glob):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            goal = str(meta.get("goal", "")).strip()
            if goal:
                goals.add(goal)
    return sorted(goals)


def pipeline_status() -> str:
    datasets = len([p for p in PROJECT_ROOT.glob("datasets/*") if p.is_dir()])
    rollouts = len([p for p in PROJECT_ROOT.glob("rollouts/*/*") if p.is_dir()])
    prefs_path = PROJECT_ROOT / "preferences" / "preferences.csv"
    prefs = 0
    if prefs_path.is_file():
        try:
            prefs = max(0, sum(1 for _ in prefs_path.open(encoding="utf-8")) - 1)
        except OSError:
            pass
    models = ", ".join(
        name for name, rel in MODEL_FILES.items() if (PROJECT_ROOT / rel).is_file()
    ) or "none"
    return (
        f"datasets: {datasets}   rollouts: {rollouts}   "
        f"preferences: {prefs}   models: {models}"
    )


class PipelineUI(tk.Tk):
    POLL_MS = 80

    def __init__(self) -> None:
        super().__init__()
        self.title("Valheim RLHF Pipeline")
        self.geometry("1020x760")
        self.minsize(820, 600)

        self.process: subprocess.Popen | None = None
        self.output_queue: queue.Queue[str | None] = queue.Queue()
        self.stages = build_stages()
        self.goal_boxes: list[ttk.Combobox] = []

        self._build_layout()
        self.refresh_status()
        self.after(self.POLL_MS, self._poll_output)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ----- layout -----------------------------------------------------------

    def _build_layout(self) -> None:
        status_bar = ttk.Frame(self, padding=(8, 6))
        status_bar.pack(fill="x")
        self.status_label = ttk.Label(status_bar, text="")
        self.status_label.pack(side="left")
        ttk.Button(status_bar, text="Refresh", command=self.refresh_status).pack(side="right")

        paned = ttk.PanedWindow(self, orient="vertical")
        paned.pack(fill="both", expand=True)

        self.notebook = ttk.Notebook(paned)
        paned.add(self.notebook, weight=3)
        for stage in self.stages:
            self.notebook.add(self._build_stage_tab(stage), text=stage.title)

        console_frame = ttk.Frame(paned, padding=(8, 4))
        paned.add(console_frame, weight=2)

        toolbar = ttk.Frame(console_frame)
        toolbar.pack(fill="x")
        self.run_state_label = ttk.Label(toolbar, text="idle", foreground="gray")
        self.run_state_label.pack(side="left")
        ttk.Button(toolbar, text="Clear", command=self._clear_console).pack(side="right")
        self.stop_button = ttk.Button(toolbar, text="Stop", command=self.stop_process, state="disabled")
        self.stop_button.pack(side="right", padx=(0, 6))

        self.console = ScrolledText(console_frame, height=12, state="disabled", wrap="word",
                                    font=("Consolas", 9))
        self.console.pack(fill="both", expand=True, pady=(4, 4))
        self.console.tag_configure("cmd", foreground="#0066cc")
        self.console.tag_configure("err", foreground="#cc0000")

        input_row = ttk.Frame(console_frame)
        input_row.pack(fill="x")
        ttk.Label(input_row, text="stdin:").pack(side="left")
        self.stdin_entry = ttk.Entry(input_row)
        self.stdin_entry.pack(side="left", fill="x", expand=True, padx=6)
        self.stdin_entry.bind("<Return>", lambda _e: self._send_stdin())
        ttk.Button(input_row, text="Send", command=self._send_stdin).pack(side="left")

    def _build_stage_tab(self, stage: Stage) -> ttk.Frame:
        tab = ttk.Frame(self.notebook, padding=10)
        ttk.Label(tab, text=stage.blurb, wraplength=920, justify="left").grid(
            row=0, column=0, columnspan=4, sticky="w", pady=(0, 8)
        )

        row = 1
        for f in stage.fields:
            ttk.Label(tab, text=f.label).grid(row=row, column=0, sticky="w", pady=2)
            if f.kind == "bool":
                f.var = tk.BooleanVar(value=f.default == "1")
                ttk.Checkbutton(tab, variable=f.var).grid(row=row, column=1, sticky="w", padx=8)
            elif f.kind == "choice":
                f.var = tk.StringVar(value=f.default)
                ttk.Combobox(tab, textvariable=f.var, values=list(f.choices),
                             state="readonly", width=22).grid(row=row, column=1, sticky="w", padx=8)
            elif f.kind == "goal":
                f.var = tk.StringVar(value=f.default)
                box = ttk.Combobox(tab, textvariable=f.var, values=DEFAULT_GOALS, width=22)
                box.grid(row=row, column=1, sticky="w", padx=8)
                self.goal_boxes.append(box)
            else:
                f.var = tk.StringVar(value=f.default)
                ttk.Entry(tab, textvariable=f.var, width=32).grid(row=row, column=1, sticky="w", padx=8)
            row += 1

        if stage.note:
            ttk.Label(tab, text=stage.note, wraplength=920, justify="left",
                      foreground="#806000").grid(row=row, column=0, columnspan=4,
                                                 sticky="w", pady=(8, 0))
            row += 1

        ttk.Button(tab, text=f"Run {stage.script}",
                   command=lambda s=stage: self.run_stage(s)).grid(
            row=row, column=0, sticky="w", pady=(12, 0)
        )
        return tab

    # ----- status -----------------------------------------------------------

    def refresh_status(self) -> None:
        self.status_label.config(text=pipeline_status())
        goals = scan_goals()
        for box in self.goal_boxes:
            box.configure(values=goals)

    # ----- process control --------------------------------------------------

    def run_stage(self, stage: Stage) -> None:
        if self.process is not None:
            messagebox.showwarning(
                "Already running",
                "Another stage is still running. Stop it before starting a new one.",
            )
            return

        cmd = [sys.executable, "-u", stage.script]
        for f in stage.fields:
            if f.kind == "bool":
                if f.var.get():
                    cmd.append(f.flag)
                continue
            value = str(f.var.get()).strip()
            if value:
                cmd.extend([f.flag, value])

        self._append(f"$ {subprocess.list2cmdline(cmd)}\n", "cmd")
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        try:
            self.process = subprocess.Popen(
                cmd,
                cwd=PROJECT_ROOT,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=creationflags,
            )
        except OSError as exc:
            self.process = None
            self._append(f"failed to start: {exc}\n", "err")
            return

        self.run_state_label.config(text=f"running: {stage.script}", foreground="#007700")
        self.stop_button.config(state="normal")
        threading.Thread(target=self._read_output, args=(self.process,), daemon=True).start()

    def _read_output(self, proc: subprocess.Popen) -> None:
        for line in proc.stdout:
            self.output_queue.put(line)
        proc.wait()
        self.output_queue.put(None)  # sentinel: process exited

    def stop_process(self) -> None:
        proc = self.process
        if proc is None:
            return
        self._append("stopping...\n", "cmd")
        try:
            if os.name == "nt":
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                proc.send_signal(signal.SIGINT)
        except OSError:
            pass
        # Escalate if it ignores the break signal.
        self.after(3000, lambda: self._force_kill(proc))

    def _force_kill(self, proc: subprocess.Popen) -> None:
        if proc.poll() is None:
            proc.terminate()

    def _send_stdin(self) -> None:
        proc = self.process
        text = self.stdin_entry.get()
        self.stdin_entry.delete(0, "end")
        if proc is None or proc.stdin is None:
            self._append("no running process to send input to\n", "err")
            return
        try:
            proc.stdin.write(text + "\n")
            proc.stdin.flush()
            self._append(f"> {text}\n", "cmd")
        except OSError as exc:
            self._append(f"stdin write failed: {exc}\n", "err")

    # ----- console ----------------------------------------------------------

    def _poll_output(self) -> None:
        try:
            while True:
                item = self.output_queue.get_nowait()
                if item is None:
                    code = self.process.returncode if self.process else "?"
                    tag = "cmd" if code == 0 else "err"
                    self._append(f"[exit code {code}]\n\n", tag)
                    self.process = None
                    self.run_state_label.config(text="idle", foreground="gray")
                    self.stop_button.config(state="disabled")
                    self.refresh_status()
                else:
                    self._append(item)
        except queue.Empty:
            pass
        self.after(self.POLL_MS, self._poll_output)

    def _append(self, text: str, tag: str | None = None) -> None:
        self.console.configure(state="normal")
        self.console.insert("end", text, tag or ())
        self.console.see("end")
        self.console.configure(state="disabled")

    def _clear_console(self) -> None:
        self.console.configure(state="normal")
        self.console.delete("1.0", "end")
        self.console.configure(state="disabled")

    def _on_close(self) -> None:
        if self.process is not None:
            if not messagebox.askyesno("Quit", "A stage is still running. Stop it and quit?"):
                return
            try:
                self.process.terminate()
            except OSError:
                pass
        self.destroy()


def main() -> int:
    app = PipelineUI()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
