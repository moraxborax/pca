from DFRobot_RaspberryPi_Motor.DFRobot_RaspberryPi_DC_Motor import (
    DFRobot_DC_Motor_IIC as Board,
)
import time
# import gpiozero

MotorBoard = Board

def board_detect(board: Board):
    l = board.detecte()
    print("Board list conform:")
    print(l)


""" print last operate status, users can use this variable to determine the result of a function call. """


def print_board_status(board: Board):
    if board.last_operate_status == board.STA_OK:
        print("board status: everything ok")
    elif board.last_operate_status == board.STA_ERR:
        print("board status: unexpected error")
    elif board.last_operate_status == board.STA_ERR_DEVICE_NOT_DETECTED:
        print("board status: device not detected")
    elif board.last_operate_status == board.STA_ERR_PARAMETER:
        print("board status: parameter error, last operate no effective")
    elif board.last_operate_status == board.STA_ERR_SOFT_VERSION:
        print("board status: unsupport board framware version")

        while board.begin() != board.STA_OK:  # Board begin and check board status
            print_board_status()
            print("board begin faild")
            time.sleep(2)
        print("board begin success")


def init_board(
    board_address: int = 0x10, reduction_ratio: int = 43, pcm_freq: int = 1000
):

    # We will ONLY use raspi
    board = Board(1, board_address)  # RaspberryPi select bus 1, set address to 0x10
    board.set_encoder_enable(board.ALL)  # Set selected DC motor encoder enable
    # board.set_encoder_disable(board.ALL)              # Set selected DC motor encoder disable
    board.set_encoder_reduction_ratio(
        board.ALL, reduction_ratio
    )  # Set selected DC motor encoder reduction ratio, test motor reduction ratio is 43.8

    board.set_moter_pwm_frequency(pcm_freq)
    return board


def forward(board: Board, duty: int):
    board.motor_movement([board.M1], board.CW, duty)
    board.motor_movement([board.M2], board.CCW, duty)


def stop(board: Board):
    forward(board, 0)


def left_inplace(board: Board, duty: int):
    board.motor_movement([board.M1], board.CCW, duty)
    board.motor_movement([board.M2], board.CCW, duty)


def right_inplace(board: Board, duty: int):
    board.motor_movement([board.M1], board.CW, duty)
    board.motor_movement([board.M2], board.CW, duty)

def forward_for(board: Board, duty: int, duration: float, dt: float=0.1):
    for _ in range(int(duration / dt)):
        forward(board, duty)
        time.sleep(dt)
    stop(board)

def left_for(board: Board, duty: int, duration: float, dt: float=0.1):
    for _ in range(int(duration / dt)):
        left_inplace(board, duty)
        time.sleep(dt)
    stop(board)

def right_for(board: Board, duty: int, duration: float, dt: float=0.1):
    for _ in range(int(duration / dt)):
        right_inplace(board, duty)
        time.sleep(dt)
    stop(board)

    # start_time = time.time()
    # while time.time() - start_time < duration:
    #     forward(board, duty)
    # stop(board)


# def search(board: Board, sonar: gpiozero.DistanceSensor):
def search(board: Board):
    # a simple search
    # just turn left slowly until you found something
    left_inplace(board, 10)

def left(board: Board, duty: int, width: float, radius: float):
    if radius < width / 2:
        raise ValueError("Radius must be greater than width / 2")
    right_duty = duty
    left_duty = duty * (radius - width / 2) / (radius + width / 2)
    board.motor_movement([board.M1], board.CW, left_duty)
    board.motor_movement([board.M2], board.CCW, right_duty)

def right(board: Board, duty: int, width: float, radius: float):
    if radius < width / 2:
        raise ValueError("Radius must be greater than width / 2")
    left_duty = duty
    right_duty = duty * (radius - width / 2) / (radius + width / 2)
    board.motor_movement([board.M1], board.CW, left_duty)
    board.motor_movement([board.M2], board.CCW, right_duty)
    


