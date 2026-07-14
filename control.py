from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# Duty cycles 0–100 for keyboard / commanded states
DRIVE_DUTY = 40
TURN_DUTY = 30
SEARCH_DUTY = 15


def _load_board_class() -> type:
    from DFRobot_RaspberryPi_Motor.DFRobot_RaspberryPi_DC_Motor import (
        DFRobot_DC_Motor_IIC as Board,
    )

    return Board


def print_board_status(board: Any) -> None:
    status = board.last_operate_status
    messages = {
        board.STA_OK: "everything ok",
        board.STA_ERR: "unexpected error",
        board.STA_ERR_DEVICE_NOT_DETECTED: "device not detected",
        board.STA_ERR_PARAMETER: "parameter error, last operate no effective",
        board.STA_ERR_SOFT_VERSION: "unsupported board firmware version",
    }
    logger.info("board status: %s", messages.get(status, f"unknown ({status})"))


def init_board(
    bus: int = 3,
    board_address: int = 0x10,
    reduction_ratio: int = 43,
    pwm_freq: int = 1000,
    retries: int = 3,
) -> Any:
    """Initialize motor board. Raises if board cannot be opened."""
    Board = _load_board_class()
    # Bus 3: Pi I2C remapped for DLP2000EVM projector hardware
    board = Board(bus, board_address)

    for attempt in range(1, retries + 1):
        if board.begin() == board.STA_OK:
            break
        print_board_status(board)
        if attempt == retries:
            raise RuntimeError("Motor board begin failed")
        logger.warning("board begin failed (%d/%d) — retrying", attempt, retries)
        time.sleep(1)
    logger.info("board begin success")

    board.set_encoder_enable(board.ALL)
    board.set_encoder_reduction_ratio(board.ALL, reduction_ratio)
    board.set_moter_pwm_frequency(pwm_freq)
    board.motor_stop(board.ALL)
    return board


def forward(board: Any, duty: int) -> None:
    board.motor_movement([board.M1], board.CCW, duty)
    board.motor_movement([board.M2], board.CCW, duty)


def stop(board: Any) -> None:
    board.motor_stop(board.ALL)


def left_inplace(board: Any, duty: int) -> None:
    board.motor_movement([board.M1], board.CW, duty)
    board.motor_movement([board.M2], board.CCW, duty)


def right_inplace(board: Any, duty: int) -> None:
    board.motor_movement([board.M1], board.CCW, duty)
    board.motor_movement([board.M2], board.CW, duty)


def search(board: Any, duty: int = SEARCH_DUTY) -> None:
    # Slow CCW spin (in-place left)
    left_inplace(board, duty)


def apply_action(
    board: Any | None,
    action: str,
    *,
    drive_duty: int = DRIVE_DUTY,
    turn_duty: int = TURN_DUTY,
    search_duty: int = SEARCH_DUTY,
) -> None:
    """Map cart class name to motor commands. No-op if board is unavailable."""
    if board is None:
        return

    if action == "forward":
        forward(board, drive_duty)
    elif action == "stop":
        stop(board)
    elif action == "left":
        left_inplace(board, turn_duty)
    elif action == "right":
        right_inplace(board, turn_duty)
    elif action == "search":
        search(board, search_duty)
    else:
        raise ValueError(f"Unknown action: {action}")
