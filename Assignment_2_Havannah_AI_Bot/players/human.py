import os
import sys
import signal
import select
import numpy as np
from typing import Tuple
from multiprocessing import Value
from helper import get_valid_actions, fetch_remaining_time, HEXAGON_COORDS, CLICK_EVENT


class HumanPlayer:
    def __init__(self, player_number, timer):
        self.player_number = player_number
        self.type = 'human'
        self.player_string = 'Player {}: human'.format(player_number)
        self.TLE_MOVE = (-1, -1)
        self.timer = timer

    @staticmethod
    def get_action(inp: str) -> Tuple[int, int]:
        action = (int(inp[0]), int(inp[1]))
        return action

    def readline_with_timeout(self, move, time_limit):
        ready, _, _ = select.select([sys.stdin], [], [], time_limit)
        if ready:
            inp = sys.stdin.readline().strip()  # Read input if available within the time limit
            move[0].value = int(inp.split(',')[0])
            move[1].value = int(inp.split(',')[1])
        else:
            move[0].value = self.TLE_MOVE[0]
            move[1].value = self.TLE_MOVE[1]

    def readLine(self, move) -> str:    
        inp = sys.stdin.readline()
        inp = inp.replace('\n', '')
        move[0].value = int(inp.split(',')[0])
        move[1].value = int(inp.split(',')[1])

    def get_input(self, time) -> str:
        print('Enter your move: ')
        move = (Value('i', -2), Value('i', -2))
        pid = os.fork()
        assert pid >= 0
        if pid == 0:
            self.readline_with_timeout(move, time)
            sys.exit()
        else:
            while not (CLICK_EVENT[0] or move[0].value >= -1):
                continue

            try:
                os.kill(pid, signal.SIGKILL)
            except:
                None

            if CLICK_EVENT[0]:
                polygon_id = CLICK_EVENT[0].widget.find_withtag("current")[0]  # Get the polygon ID
                move = HEXAGON_COORDS[polygon_id]
                print(move)
                CLICK_EVENT[0] = False
                return move

            return move[0].value, move[1].value

    def get_move(self, state: Tuple[np.array]) -> Tuple[int, int]:
        """
        Given the current state returns the next action

        # Parameters
        `state: Tuple[np.array]`
            - a numpy array containing the state of the board using the following encoding:
            - the board maintains its same two dimensions
            - spaces that are unoccupied are marked as 0
            - spaces that are blocked are marked as 3
            - spaces that are occupied by player 1 have a 1 in them
            - spaces that are occupied by player 2 have a 2 in them
        
        # Returns
        Tuple[int, int]: action (coordinates of a board cell)
        """
        valid_actions = get_valid_actions(state, self.player_number)
        action = self.get_action(self.get_input(fetch_remaining_time(self.timer, self.player_number)))
        if action == self.TLE_MOVE:
            print('Time Limit Exceeded')
        elif action not in valid_actions:
            print('Invalid Move: Choose from: {}'.format(valid_actions))
            print('Turning to other player')
            print("ACTION ==", action)
        return action
