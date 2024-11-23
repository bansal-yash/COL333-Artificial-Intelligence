import random
import numpy as np
from time import sleep
from typing import Tuple
from helper import get_valid_actions, fetch_remaining_time


class RandomPlayer:
    def __init__(self, player_number, timer):
        self.player_number = player_number
        self.type = 'random'
        self.player_string = 'Player {}: random'.format(player_number)
        self.timer = timer

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
        sleep(0.01)
        valid_actions = get_valid_actions(state, self.player_number)
        action        = random.choice(valid_actions)
        return int(action[0]), int(action[1])
