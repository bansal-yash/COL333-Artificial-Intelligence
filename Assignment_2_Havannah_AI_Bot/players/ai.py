import time
import math
import random
import numpy as np
from helper import *
from collections import defaultdict
from typing import List, Tuple, Dict, Union


C = 0.9
C_OPP = 0.9
INF = 100000
# Load the start board, if ai is player1 then load empty board, otherwise load the board after the first move of the opponent
start_board = 0


# Matrix to String
def node_to_str(node_copy : np.array):
    node = node_copy.copy()
    node = np.matrix(node_copy)
    num_rows, num_cols = node.shape
    dim = (num_rows+1)//2
    row_len = 0
    if(dim == 4):
        row_len = 5
    else:
        row_len = 7
    full_str = ''
    for i in range(num_rows):
        temp_num = 0
        temp_str = ''
        for j in range(num_cols):
            temp_num += int(node[i, j]) * (4**(j))
        temp_str = str(temp_num)
        while(len(temp_str) != row_len):
            temp_str = '0' + temp_str
        full_str = temp_str + full_str
    return full_str

# String to matrix
def str_to_node(s: str, num_rows, num_cols):
    matrix = np.zeros((num_rows, num_cols), dtype=int)
    temp_num = 0
    temp_str = ''
    dim = (num_rows + 1)//2
    row_len = 0
    if(dim == 4):
        row_len = 5
    else:
        row_len = 7
    for i in range(num_rows):
        temp_str = s[-row_len:]
        s = s[:-row_len]
        temp_num = int(temp_str)
        for j in range(num_cols):
            matrix[i, j] = temp_num % 4
            temp_num //= 4
    return matrix

def find_action(old_arr : np.array, new_arr : np.array):
    num_rows, num_cols = old_arr.shape
    for row in range(num_rows):
        for col in range(num_cols):
            if(old_arr[row][col] != new_arr[row][col]):
                return (row, col)

def find_first_one(arr: np.array):
    result = np.argwhere(arr == 1)
    if result.size > 0:
        return tuple(result[0])
    else:
        return None

def can_player_win_in_one_move_1(state, all_valid_moves, player):
    new_state = state.copy()
    dim = state.shape[0]

    for move in all_valid_moves:
        neighbours = get_neighbours(dim, move)
        to_check = False

        for neighbour in neighbours:
            if new_state[neighbour] == player:
                to_check = True
                break

        if to_check:
            new_state[move] = player
            bool_state = (new_state == player)
            if ( check_ring(bool_state, move)  or check_bridge(bool_state, move) or check_fork(bool_state, move) ):
                return move
            new_state[move] = 0
    
    return False

class MCTS:
    def __init__(self, player_num, opponent_num, state, nodes_to_check):
        self.explored_nodes = []
        self.children_nodes = defaultdict(int)
        self.num_wins = defaultdict(int)
        self.num_visits = defaultdict(int)
        self.player_num = player_num
        self.opponent_num = opponent_num
        self.num_rows = state.shape[0]
        self.root_state = node_to_str(state)
        self.root_to_expand = nodes_to_check

    def is_leaf_node(self, node: np.array, chosen_move = None, player_num = None):
        if chosen_move != None:
            if player_num == self.player_num:
                bool_state = (node == self.player_num)

                if(check_bridge(bool_state, chosen_move) or check_fork(bool_state, chosen_move) or check_ring(bool_state, chosen_move)):
                    return True, 1
                
            if player_num == self.opponent_num:
                bool_state = (node == self.opponent_num)
                if(check_bridge(bool_state, chosen_move) or check_fork(bool_state, chosen_move) or check_ring(bool_state, chosen_move)):
                    return True, -1
            
            return False, 0
        
        else:
            return False, 0

    def run_simulation(self, node : np.array, node_num : str):
        player_num = self.player_num
        num_rows, num_cols = node.shape
        path, player_num = self.select(node_num, player_num)
        unexplored_descendent_num = path[-1]
        unexplored_descendent_node = str_to_node(unexplored_descendent_num, num_rows, num_cols)
        self.expand(unexplored_descendent_node, unexplored_descendent_num, player_num)
        if len(path) == 1:
            is_win = self.simulate(unexplored_descendent_node, player_num)
        else:
            is_win = self.simulate(unexplored_descendent_node, player_num, path[-2])
        self.back_propagate(is_win, path)

    def get_children_keys(self, node : np.array, player_num : int):
        children_keys = []
        if (node_to_str(node) == self.root_state):
            return self.root_to_expand
        
        node_copy = node.copy()
        valid_moves = get_valid_actions(node_copy, player_num)
        for move in valid_moves:
            node_copy[move] = player_num
            node_num = node_to_str(node_copy)
            children_keys.append(node_num)
            node_copy[move] = node[move]
        return children_keys
    
    def rollout_policy(self, node : np.array ,  all_moves, player_num):
        # for now return a random move
        # remember to delete the move from the  all_moves array also
        # res = can_player_win_in_one_move_1(node, all_moves, player_num)
        # if (res != False):
        #     all_moves.remove(res)
        #     return res
        
        # if(player_num == 1): opp_player = 2
        # else: opp_player = 1

        # res = can_player_win_in_one_move_1(node, all_moves, opp_player)
        # if (res != False):
        #     all_moves.remove(res)
        #     return res

        selected_move = random.choice(all_moves)
        all_moves.remove(selected_move)
        return selected_move
    
    def pop_from(self, children_list : List[str]):
        selected_child = random.choice(children_list)
        return selected_child

    def uct_select(self, parent_key, children_keys, player_num, nodes_to_check = None, best_flag = False):
        # selected_child = random.choice(children_keys)
        # return selected_child
        max_uct_val = -INF
        max_uct_node = ''
        uct_val = 0

        c_val = C
        c_opp_val = C_OPP
        if best_flag:
            c_val = 0

        parent_log = math.log(self.num_visits[parent_key])
        for child in children_keys:
            if (nodes_to_check != None and len(nodes_to_check) != 0):
                if child not in nodes_to_check:
                    continue
            
            if(self.num_visits[child] == 0):
                uct_val = INF
                return child
            else:
                if player_num == self.player_num:
                    uct_val = (self.num_wins[child]/self.num_visits[child]) + c_val*(math.sqrt( parent_log/ self.num_visits[child]))
                else:
                    uct_val = (self.num_wins[child]/self.num_visits[child]) + c_opp_val*(math.sqrt( parent_log/ self.num_visits[child]))
                
            if(uct_val >= max_uct_val):
                max_uct_val = uct_val
                max_uct_node = child
        return max_uct_node
    
    def select(self, node_num : str, player_num : int):
        # this function will return the path to one of the descendents of node which is not explored
        path = [node_num]

        while True:
            if(node_num not in self.explored_nodes): # node is unexplored
                return path, player_num
            
            if(not self.children_nodes[node_num]): # node is explored, but is terminal with no children
                return path, player_num

            children_keys = self.children_nodes[node_num]
            unexplored_children = [item for item in children_keys if item not in self.explored_nodes]
            if(len(unexplored_children) != 0):
                nxt_node_num = self.pop_from(unexplored_children)
                path.append(nxt_node_num)

                if(player_num == 1): player_num = 2
                else: player_num = 1

                return path, player_num
            
            else:
                node_num = self.uct_select(node_num, children_keys, player_num)
                path.append(node_num)

                if(player_num == 1): player_num = 2
                else: player_num = 1

    def expand(self, node : np.array, node_num : str, player_num : int):
        # a node is expandid only when it is unexplored. so add the node to explored nodes
        if node_num in self.explored_nodes:
            return
        else:
            self.explored_nodes.append(node_num)
            self.children_nodes[node_num] = self.get_children_keys(node, player_num)

    def simulate(self, node : np.array, player_num : int, par_str = None):
        """This function takes the node, and knows that this node is expanded already. From all of these expanded children of the nodes
        It will select one randomly and then simulate random moves until leaf is attained"""
        curr_state = node.copy()
        if par_str == None:
            is_leaf, is_win = self.is_leaf_node(curr_state)
        else:
            par_node = str_to_node(par_str, self.num_rows, self.num_rows)
            act = find_action(par_node, curr_state)
            is_leaf, is_win = self.is_leaf_node(curr_state, act, player_num)
        all_possible_moves = get_valid_actions(curr_state)
        chosen_move = 0

        while(is_leaf == False and len(all_possible_moves) != 0):
            # Choose a move to take according to a rollout policy
            chosen_move = self.rollout_policy(curr_state, all_possible_moves, player_num) # Heuristic 1
            # Make the chosen move and remove the chosen_move from the all_possible_moves array
            curr_state[chosen_move[0], chosen_move[1]] = player_num
            # Change the chance of the player for the next move
            # Make a check if leaf is reached
            is_leaf, is_win = self.is_leaf_node(curr_state, chosen_move, player_num)

            if(player_num == 1): player_num = 2
            else: player_num = 1

            if(len(all_possible_moves) == 0):
                break
        return is_win

    def back_propagate(self, is_win, path : List[str]):
        is_win = (-1) * is_win
        for node_num in path:
            self.num_visits[node_num] += 1
            self.num_wins[node_num] += is_win
            is_win = (-1)*is_win

    def choose_best_node(self, node : np.array, node_num : str, nodes_to_check):
        # is_leaf, is_win = self.is_leaf_node(node)

        children_list = self.get_children_keys(node, self.player_num)
        if(node_num not in self.explored_nodes):
            return self.pop_from(children_list)
        max_uct_node = self.uct_select(node_num, children_list, self.player_num, nodes_to_check, True)
        return max_uct_node


"""
##################################################################################################################
##################################################################################################################
##################################################################################################################
"""


class AIPlayer:

    def __init__(self, player_number: int, timer):
        """
        Intitialize the AIPlayer Agent

        # Parameters
        `player_number (int)`: Current player number, num==1 starts the game
        
        `timer: Timer`
            - a Timer object that can be used to fetch the remaining time for any player
            - Run `fetch_remaining_time(timer, player_number)` to fetch remaining time of a player
        """
        self.player_number = player_number
        self.opponent_player_number = (1 if player_number == 2 else 2)
        self.type = 'ai'
        self.player_string = 'Player {}: ai'.format(player_number)
        self.timer = timer
        self.num_moves = 0
        self.tree = None
        self.curr_board = None
        self.mcts_time = 9
        self.first_move = None
        self.initial_time = fetch_remaining_time(timer, self.player_number)

    def can_player_win_in_one_move(self, state, all_valid_moves, player):
        new_state = state.copy()
        dim = state.shape[0]

        for move in all_valid_moves:
            neighbours = get_neighbours(dim, move)
            to_check = False

            for neighbour in neighbours:
                if new_state[neighbour] == player:
                    to_check = True
                    break

            if to_check:
                new_state[move] = player
                bool_state = (new_state == player)
                if ( check_ring(bool_state, move)  or check_bridge(bool_state, move) or check_fork(bool_state, move) ):
                    return move
                new_state[move] = 0
        
        return False

    def can_player_win_in_two_moves(self, state, all_valid_moves, player):
        new_state = state.copy()
        dim = state.shape[0]
        can_threaten = []

        for move_1 in all_valid_moves:
            new_state[move_1] = player
            move_adv = 0
            reachable_nodes = bfs_reachable(new_state == player, move_1)
            neigh_set = set()

            for n in reachable_nodes:
                neigh = get_neighbours(dim, n)
                for a in neigh:
                    if (new_state[a] == 0):
                        neigh_set.add(a)

            for move_2 in neigh_set:
                new_state[move_2] = player
                bool_state = (new_state == player)
                if ( check_ring(bool_state, move_2)  or check_bridge(bool_state, move_2) or check_fork(bool_state, move_2) ):
                    move_adv += 1
                    if move_adv > 1:
                        return move_1
                new_state[move_2] = 0

            if move_adv == 1:
                can_threaten.append(move_1)
            new_state[move_1] = 0
    
        return (100, can_threaten)
    
    def can_player_win_in_three_moves(self, state, all_valid_moves, player, can_threaten):
        new_state = state.copy()
        dim = state.shape[0]

        if len(can_threaten) == 0:
            for move_1 in all_valid_moves:
                new_state[move_1] = player
                new_valid_moves = all_valid_moves.copy()
                new_valid_moves.remove(move_1)

                reachable_nodes = bfs_reachable(new_state == player, move_1)
                neigh_set = set()

                for n in reachable_nodes:
                    neigh = get_neighbours(dim, n)
                    for a in neigh:
                        if (new_state[a] == 0):
                            neigh_set.add(a)

                move_upar_adv = 0

                for move_2 in neigh_set:
                    move_adv = 0
                    new_state[move_2] = player
                    neigh_set_1 = neigh_set.copy()
                    neigh_1 = get_neighbours(dim, move_2)
                    for a1 in neigh_1:
                        if (new_state[a1] == 0 and a1 != move_1):
                            neigh_set_1.add(a1)
                    neigh_set_1.remove(move_2)

                    for move_3 in neigh_set_1:
                        new_state[move_3] = player
                        bool_state = (new_state == player)
                        if ( check_ring(bool_state, move_3)  or check_bridge(bool_state, move_3) or check_fork(bool_state, move_3) ):
                            move_adv += 1
                        new_state[move_3] = 0
                        if move_adv > 1:
                            break
                    
                    if move_adv > 1:
                        move_upar_adv += 1
                        if move_upar_adv > 1:
                            return move_1
                        
                    new_state[move_2] = 0

                new_state[move_1] = 0
            
        else:
            for move_1 in can_threaten:
                new_state[move_1] = player
                new_valid_moves = all_valid_moves.copy()
                new_valid_moves.remove(move_1)

                reachable_nodes = bfs_reachable(new_state == player, move_1)
                neigh_set = set()
                move_upar_adv = 0

                for n in reachable_nodes:
                    neigh = get_neighbours(dim, n)
                    for a in neigh:
                        if (new_state[a] == 0):
                            neigh_set.add(a)

                for move_2 in neigh_set:
                    move_adv = 0
                    new_state[move_2] = player
                    neigh_set_1 = neigh_set.copy()
                    neigh_1 = get_neighbours(dim, move_2)
                    for a1 in neigh_1:
                        if (new_state[a1] == 0 and a1 != move_1):
                            neigh_set_1.add(a1)
                    neigh_set_1.remove(move_2)

                    for move_3 in neigh_set_1:
                        new_state[move_3] = player
                        bool_state = (new_state == player)
                        if ( check_ring(bool_state, move_3)  or check_bridge(bool_state, move_3) or check_fork(bool_state, move_3) ):
                            move_adv += 1

                        new_state[move_3] = 0
                        if move_adv > 1:
                            break;            
                    new_state[move_2] = 0

                    if move_adv > 1:
                        move_upar_adv += 1
                        if move_upar_adv > 1:
                            return move_1

                new_state[move_1] = 0    
        
        return False
    
    def get_move(self, state: np.array) -> Tuple[int, int]:
        t1 = time.time()
        self.num_moves += 1
        """
        Given the current state of the board, return the next move

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
        # Do the rest of your implementation here

        all_valid_moves = get_valid_actions(state)
        random.shuffle(all_valid_moves)
        dim = state.shape[0]

        if dim == 7:
            self.mcts_time = 10
        else:
            if self.initial_time > 250 and self.initial_time < 350:
                if self.num_moves >= 1 and self.num_moves <= 10:
                    self.mcts_time = 9
                elif self.num_moves >= 11 and self.num_moves <= 25:
                    time_remaining = fetch_remaining_time(self.timer, self.player_number ) - 110
                    moves_remaining = 26 - self.num_moves
                    self.mcts_time = (time_remaining / moves_remaining)
                elif self.num_moves >= 26 and self.num_moves <= 35:
                    time_remaining = fetch_remaining_time(self.timer, self.player_number ) - 40
                    moves_remaining = 36 - self.num_moves
                    self.mcts_time = (time_remaining / moves_remaining)                
                elif self.num_moves >= 36:
                    time_remaining = fetch_remaining_time(self.timer, self.player_number) - 10
                    self.mcts_time = (time_remaining / 10)

            if self.initial_time > 400 and self.initial_time < 550:
                if self.num_moves >= 1 and self.num_moves <= 10:
                    self.mcts_time = 16
                elif self.num_moves >= 11 and self.num_moves <= 25:
                    time_remaining = fetch_remaining_time(self.timer, self.player_number ) - 180
                    moves_remaining = 26 - self.num_moves
                    self.mcts_time = (time_remaining / moves_remaining)
                elif self.num_moves >= 26 and self.num_moves <= 35:
                    time_remaining = fetch_remaining_time(self.timer, self.player_number ) - 70
                    moves_remaining = 36 - self.num_moves
                    self.mcts_time = (time_remaining / moves_remaining)                
                elif self.num_moves >= 36:
                    time_remaining = fetch_remaining_time(self.timer, self.player_number) - 10
                    self.mcts_time = (time_remaining / 10)

            
            if self.initial_time > 550:
                if self.num_moves >= 1 and self.num_moves <= 10:
                    self.mcts_time = 22
                elif self.num_moves >= 11 and self.num_moves <= 25:
                    time_remaining = fetch_remaining_time(self.timer, self.player_number ) - 200
                    moves_remaining = 26 - self.num_moves
                    self.mcts_time = (time_remaining / moves_remaining)
                elif self.num_moves >= 26 and self.num_moves <= 35:
                    time_remaining = fetch_remaining_time(self.timer, self.player_number ) - 70
                    moves_remaining = 36 - self.num_moves
                    self.mcts_time = (time_remaining / moves_remaining)                
                elif self.num_moves >= 36:
                    time_remaining = fetch_remaining_time(self.timer, self.player_number) - 10
                    self.mcts_time = (time_remaining / 10)

        if True:
            # Seeing confirm win/loss in one/two moves:- Play winning move/ block opponent's winning move
            res = self.can_player_win_in_one_move(state, all_valid_moves, self.player_number)
            if (res != False):
                res = (int(res[0]), int(res[1]))
                return res
                
            res = self.can_player_win_in_one_move(state, all_valid_moves, self.opponent_player_number)
            if (res != False):
                res = (int(res[0]), int(res[1]))
                return res

            res = self.can_player_win_in_two_moves(state, all_valid_moves, self.player_number)
            if (res[0] != 100):
                res = (int(res[0]), int(res[1]))
                return res
            else:
                i_can_threaten = res[1]

            res = self.can_player_win_in_two_moves(state, all_valid_moves, self.opponent_player_number)
            if (res[0] != 100):
                res = (int(res[0]), int(res[1]))
                return res
            else:
                opp_can_threaten = res[1]

            if (len(opp_can_threaten) <= 1):
                res = self.can_player_win_in_three_moves(state, all_valid_moves, self.player_number, opp_can_threaten)
                if (res != False):
                    res = (int(res[0]), int(res[1]))
                    return res
            
            if (len(i_can_threaten) <= 1):
                res = self.can_player_win_in_three_moves(state, all_valid_moves, self.opponent_player_number, i_can_threaten)
                if (res != False):
                    res = (int(res[0]), int(res[1]))
                    return res
        
        # Hardcodings in this section
        if dim == 11:
            middle_moves = [ (3,4), (4,5), (3,6), (4,4), (5,5), (4,6), (5,4), (6,5), (5,6)]
            if self.num_moves == 1:
                for a in middle_moves:
                    if state[a] == 0:
                        self.first_move = a
                        return a
            
            if self.num_moves == 2:
                neigh = get_neighbours(dim, self.first_move)
                for a in neigh:
                    if state[a] == 0:
                        return a 

            if self.num_moves <= 4:
                corners = [ (0,5), (0,0), (0,10), (5, 0), (5,10), (10, 5)]
                for move in corners:
                    if state[move] == 0:
                        move = (int(move[0]), int(move[1]))
                        return move
                    
        if dim == 7:
            corner_cases = [[(0, 0), (0, 3), (1, 2)] ,
                            [(0, 3), (0, 6), (1, 4)]  , 
                            [(0, 6), (2, 5), (3, 6)]  ,
                            [(3, 6), (4, 4), (6, 3)] ,
                            [(6, 3), (3, 0), (4, 2)] ,
                            [(3, 0), (2, 1), (0, 0)]  ]
            to_return = None
            for case in corner_cases:
                if (state[case[0]] == self.player_number and state[case[1]] == self.player_number):
                    if state[case[2]] == 0:
                        to_return = case[2]
                if (state[case[1]] == self.player_number and state[case[2]] == self.player_number):
                    if state[case[0]] == 0:
                        to_return = case[0]
                if (state[case[0]] == self.player_number and state[case[2]] == self.player_number):
                    if state[case[1]] == 0:
                        to_return = case[1]

            if to_return != None:
                to_return = ( int(to_return[0]), int(to_return[1]))
                return to_return
               
            for case in corner_cases:
                if (state[case[0]] == self.opponent_player_number and state[case[1]] == self.opponent_player_number):
                    if state[case[2]] == 0:
                        to_return = case[2]
                if (state[case[1]] == self.opponent_player_number and state[case[2]] == self.opponent_player_number):
                    if state[case[0]] == 0:
                        to_return = case[0]
                if (state[case[0]] == self.opponent_player_number and state[case[2]] == self.opponent_player_number):
                    if state[case[1]] == 0:
                        to_return = case[1]
            
            if to_return != None:
                to_return = ( int(to_return[0]), int(to_return[1]))
                return to_return         
                    
            if self.num_moves <= 3:
                corners = get_all_corners(dim)
                random.shuffle(corners)
                for move in corners:
                    if state[move] == 0:
                        move = ( int(move[0]), int(move[1]))
                        return move
                         
        # Now the code for connected components
        if True:
            player_visted = set()
            player_cc = []
            player_bool_state = (state == self.player_number)

            for i in range(dim):
                for j in range(dim):
                    if player_bool_state[i][j] == True and ((i, j) not in player_visted):
                        bfs_out = bfs_reachable(player_bool_state, (i, j))
                        player_visted = player_visted.union(bfs_out)
                        player_cc.append(bfs_out)

            player_cc_neighbour = []
            s_list = []
            for cc in player_cc:
                neighbours_of_cc = set()
                for a in cc:
                    neighbours = get_neighbours(dim, a)
                    for b in neighbours:
                        if state[b] == 0:
                            neighbours_of_cc.add(b)
                player_cc_neighbour.append(neighbours_of_cc)
                s_list.append(len(cc) + len(neighbours_of_cc))
            
            if len(s_list) == 0:
                player_neighbour = set()
            if len(s_list) == 1:
                player_neighbour = player_cc_neighbour[0]
            else:
                indices = sorted(range(len(s_list)), key=lambda i: s_list[i], reverse=True)[:2]
                player_neighbour = (player_cc_neighbour[indices[0]]).union( (player_cc_neighbour[indices[1]]))


            opp_visited = set()
            opp_cc = []
            opp_bool_state = (state == self.opponent_player_number)

            for i in range(dim):
                for j in range(dim):
                    if opp_bool_state[i][j] == True and ((i, j) not in opp_visited):
                        bfs_out = bfs_reachable(opp_bool_state, (i, j))
                        player_visted = player_visted.union(bfs_out)
                        opp_cc.append(bfs_out)

            opp_cc_neighbour = []
            s_list = []
            for cc in opp_cc:
                neighbours_of_cc = set()
                for a in cc:
                    neighbours = get_neighbours(dim, a)
                    for b in neighbours:
                        if state[b] == 0:
                            neighbours_of_cc.add(b)
                opp_cc_neighbour.append(neighbours_of_cc)
                s_list.append(len(cc) + len(neighbours_of_cc))
            
            if len(s_list) == 0:
                opp_neighbour = set()
            if len(s_list) == 1:
                opp_neighbour = opp_cc_neighbour[0]
            if len(s_list) == 2:
                opp_neighbour = (opp_cc_neighbour[0]).union(opp_cc_neighbour[1])
            else:
                indices = sorted(range(len(s_list)), key=lambda i: s_list[i], reverse=True)[:3]
                opp_neighbour = (opp_cc_neighbour[indices[0]]).union( (opp_cc_neighbour[indices[1]])).union(opp_cc_neighbour[indices[2]])

            nodes_to_check = player_neighbour.union(opp_neighbour)


        nodes_to_check_str = set()
        for a in nodes_to_check:
            new_board = state.copy()
            new_board[a] = self.player_number
            str_board = node_to_str(new_board)
            nodes_to_check_str.add(str_board)

        # Implementing MCTS now 
        # if(self.tree == None):
        if True:
            self.tree = MCTS(self.player_number, self.opponent_player_number, state, nodes_to_check_str)
            
        try:
            self.curr_board = state.copy()
            curr_board_num = node_to_str(self.curr_board)
            num_iter = 0

            while (time.time() - t1 < self.mcts_time):
                num_iter += 1
                self.tree.run_simulation(self.curr_board, curr_board_num)
            # print(num_iter)
            num_rows, num_cols = state.shape

            node_num_after_action = self.tree.choose_best_node(self.curr_board, curr_board_num, nodes_to_check_str)
            node_after_action = str_to_node(node_num_after_action, num_rows, num_cols)
            action_to_take = find_action(self.curr_board, node_after_action)
            action_to_take = ( int(action_to_take[0]), int(action_to_take[1]))
            return action_to_take
        except:
            # print("giving random move")
            all_valid_moves = get_valid_actions(state, self.player_number)
            move = random.choice(all_valid_moves)
            move = ( int(move[0]), int(move[1]) )
            return move
