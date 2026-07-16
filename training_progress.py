"""Shared training progress for live frontend polling."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock


@dataclass
class TrainingProgress:
    active: bool = False
    epoch: int = 0
    total_epochs: int = 0
    step: int = 0
    steps_per_epoch: int = 0
    total_steps: int = 0
    global_step: int = 0
    message: str = ""


_lock = Lock()
_progress = TrainingProgress()


def reset(total_epochs: int, steps_per_epoch: int) -> None:
    with _lock:
        _progress.active = True
        _progress.epoch = 0
        _progress.total_epochs = total_epochs
        _progress.step = 0
        _progress.steps_per_epoch = steps_per_epoch
        _progress.total_steps = total_epochs * steps_per_epoch
        _progress.global_step = 0
        _progress.message = "Starting training…"


def update(
    *,
    epoch: int,
    step: int,
    message: str | None = None,
) -> None:
    with _lock:
        _progress.epoch = epoch
        _progress.step = step
        _progress.global_step = (epoch - 1) * _progress.steps_per_epoch + step
        if message is not None:
            _progress.message = message


def finish(message: str = "Training complete") -> None:
    with _lock:
        _progress.active = False
        _progress.epoch = _progress.total_epochs
        _progress.step = _progress.steps_per_epoch
        _progress.global_step = _progress.total_steps
        _progress.message = message


def fail(message: str) -> None:
    with _lock:
        _progress.active = False
        _progress.message = message


def snapshot() -> dict:
    with _lock:
        p = _progress
        return {
            "active": p.active,
            "epoch": p.epoch,
            "total_epochs": p.total_epochs,
            "step": p.step,
            "steps_per_epoch": p.steps_per_epoch,
            "total_steps": p.total_steps,
            "global_step": p.global_step,
            "message": p.message,
        }
