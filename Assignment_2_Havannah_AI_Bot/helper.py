import heapq
import numpy as np
from collections import deque
from typing import List, Tuple, Dict, Union
from multiprocessing import Array


PLAYER_TIME = Array('f', [0, 0])
HEXAGON_COORDS = {}
CLICK_EVENT = [None]


def is_valid(x, y, dims):
    '''
    Returns whether the coordinates are valid or not

    # Parameters
    `x (int)`: x-coordinate
    `y (int)`: y-coordinate
    `dims (int)`: Dimension of the board

    # Returns
    bool: True if the coordinates lie inside the board, False otherwise
    '''
    return 0 <= x < dims and 0 <= y < dims

def fetch_remaining_time(timer, player_num: int) -> float:
    '''
    Returns the remaining time for the player

    # Parameters
    `timer`: Timer object
    `player_num (int)`: Player number (1 or 2)

    # Returns
    float: Remaining time for the player
    '''
    return timer[player_num - 1]

def get_valid_actions(board: np.array, player: int = None) -> List[Tuple[int, int]]:
    '''
    Returns all the valid actions in the provided state `board`
    
    # Parameters
    `board (numpy array)`: Game board

    # Returns
    List[Tuple[int]]: List of valid actions, coordinates of the valid moves
    '''
    valid_moves = np.argwhere(board == 0)
    valid_moves = [tuple(move) for move in valid_moves]
    return valid_moves

def get_vertices_on_edge(edge: int, dim: int) -> List[Tuple[int, int]]:
    '''
    Returns the vertices on an edge of the board
    
    # Parameters
    edge (int): A number from 0 to 5, representing an edge
    dim (int): Dimension of the board

    # Returns
    List[Tuple[int, int]]: List of coordinates of the vertices on the edge, if edge is valid else returns empty list
    '''
    if edge == 0:
        return [(i, 0) for i in range(1, dim // 2)]
    if edge == 1:
        return [(0, j) for j in range(1, dim // 2)]
    if edge == 2:
        return [(0, j) for j in range(dim // 2 + 1, dim - 1)]
    if edge == 3:
        return [(i, dim - 1) for i in range(1, dim // 2)]
    if edge == 4:
        return [(i, 3 * (dim // 2) - i) for i in range(dim // 2 + 1, dim - 1)]
    if edge == 5:
        return [(i, i - dim // 2) for i in range(dim // 2 + 1, dim - 1)]
    return []


def get_vetex_at_corner(corner: int, dim: int) -> Tuple[int, int]:
    '''
    Returns the vertex at a corner of the board
    
    # Parameters
    corner (int): A number from 0 to 5, representing a corner
    dim (int): Dimension of the board

    # Returns
    Tuple[int, int]: Coordinates of the vertex at the corner, if corner is valid else returns (-1, -1)
    '''
    if corner == 0:
        return 0, 0
    if corner == 1:
        return 0, dim // 2
    if corner == 2:
        return 0, dim - 1
    if corner == 3:
        return dim // 2, dim - 1
    if corner == 4:
        return dim - 1, dim // 2
    if corner == 5:
        return dim // 2, 0
    return -1, -1


def get_edge(vertex: Tuple[int, int], dim: int) -> int:
    '''
    Returns the edge on which the vertex lies
    
    # Parameters
    vertex (Tuple[int, int]): Coordinates of the point whose virtual neighbors are to be found
    dim (int): Dimension of the board

    # Returns
    int: Number of the edge on which the vertex lies, if it does else returns -1. Edges are numbered from 0 to 5
    '''
    i, j = vertex
    if j == 0 and i > 0 and i < dim // 2:
        return 0
    if i == 0 and j > 0 and j < dim // 2:
        return 1
    if i == 0 and j > dim // 2 and j < dim - 1:
        return 2
    if j == dim - 1 and i > 0 and i < dim // 2:
        return 3
    if i > dim // 2 and i < dim - 1 and i + j == 3 * (dim // 2):
        return 4
    if i > dim // 2 and i < dim - 1 and i - j == dim // 2:
        return 5
    return -1


def get_corner(vertex: Tuple[int, int], dim: int) -> int:
    '''
    Returns the corner at which the vertex lies
    
    # Parameters
    vertex (Tuple[int, int]): Coordinates of the point whose virtual neighbors are to be found
    dim (int): Dimension of the board

    # Returns
    int: Number of the corner at which the vertex lies, if it does else returns -1. Corners are numbered from 0 to 5
    '''
    i, j = vertex
    if i == 0 and j == 0:
        return 0
    if i == 0 and j == dim // 2:
        return 1
    if i == 0 and j == dim - 1:
        return 2
    if i == dim // 2 and j == dim - 1:
        return 3
    if i == dim - 1 and j == dim // 2:
        return 4
    if i == dim // 2 and j == 0:
        return 5
    return -1


def get_neighbours(dim: int, vertex: Tuple[int, int]) -> List[Tuple[int, int]]:
    '''
    Returns the neighbours of the vertex on the board
    
    # Parameters
    dim (int): Dimension of the board
    vertex (Tuple[int, int]): Coordinates of the point whose virtual neighbors are to be found

    # Returns
    List[List[Tuple[int, int]]]: List of list of tuples, where each list contains 3 vertices.
        - First tuple in each list is a virtual neighbour of the "vertex"
        - Second and Third tuples in each list are common neighbours of the "vertex" and the virtual neighbour
    '''
    i, j = vertex
    siz = dim //2
    neighbours = []
    if i > 0:
        neighbours.append((i - 1, j))
    if i < dim - 1:
        neighbours.append((i + 1, j))
    if j > 0:
        neighbours.append((i, j - 1))
    if j < dim - 1:
        neighbours.append((i, j + 1))
    if i > 0 and j <= siz and j > 0:
        neighbours.append((i - 1, j - 1))
    if i > 0 and j >= siz and j < dim - 1:
        neighbours.append((i - 1, j + 1))
    if j < siz and i < dim - 1:
        neighbours.append((i + 1, j + 1))
    if j > siz and i < dim - 1:
        neighbours.append((i + 1, j - 1))
    return neighbours

def get_all_corners(dim: int) -> List[Tuple[int, int]]:
    '''
    Returns vertices on all the corners of the board
    
    # Parameters
    dim (int): Dimension of the board

    # Returns
    List[Tuple[int, int]]: List containing the coordinates of the corner vertices of the board as tuples
    '''
    return [(0, 0),
            (0, dim // 2),
            (0, dim - 1),
            (dim // 2, dim - 1),
            (dim - 1, dim // 2),
            (dim // 2, 0)]

def get_all_edges(dim: int) -> List[List[Tuple[int, int]]]:
    '''
    Returns vertices on all the edges of the board
    
    # Parameters
    dim (int): Dimension of the board

    # Returns
    List[Tuple[int, int]]: List containing the coordinates of the edge vertices of the board as tuples
    '''
    siz = ((dim+1)//2)
    sides = [  # coordinates of the sides of the board (not including the corners)
        [(0, i) for i in range(1, siz-1)],
        [(0, i) for i in range(siz, dim-1)],
        [(i, dim-1) for i in range(1, siz-1)],
        [(siz-1+i, dim-1-i) for i in range(1, siz-1)],
        [(siz-1+i, i) for i in range(1, siz-1)],
        [(i, 0) for i in range(1, siz-1)]
    ]
    return sides


def move_coordinates(direction: str, half: int) -> Tuple[int, int]:
    '''
    Returns the coordinates of the move in the given direction
    
    # Parameters
    `direction (str)`: The direction to which the move is to be made
    `half (int)`: The half of the board from which the move is to be made.
        - half = 0 => mid-line
        - half < 0 => left half
        - half > 0 => right half

    # Returns
    Tuple[int, int]: Coordinates of the move in the given direction
    '''
    if direction == "up":
        return (-1, 0)
    elif direction == "down":
        return (1, 0)
    elif direction == "top-left":
        if half == 0:
            return (-1, -1)
        elif half < 0:
            return (-1, -1)
        elif half > 0:
            return (0, -1)
    elif direction == "top-right":
        if half == 0:
            return (-1, 1)
        elif half < 0:
            return (0, 1)
        elif half > 0:
            return (-1, 1)
    elif direction == "bottom-left":
        if half == 0:
            return (0, -1)
        elif half < 0:
            return (0, -1)
        elif half > 0:
            return (1, -1)
    elif direction == "bottom-right":
        if half == 0:
            return (0, 1)
        elif half < 0:
            return (1, 1)
        elif half > 0:
            return (0, 1)

    return None


def three_forward_moves(direction: str) -> List[str]:
    '''
    Returns the 3 forward moves from the current direction
    
    # Parameters
    direction (str): The direction of the last move

    # Returns
    List[str]: List of 3 forward moves from the current direction
    '''
    if direction == "up":
        return ["top-left", "up", "top-right"]
    if direction == "down":
        return ["down", "bottom-left", "bottom-right"]
    if direction == "top-left":
        return ["bottom-left", "top-left", "up"]
    if direction == "top-right":
        return ["top-right", "up", "bottom-right"]
    if direction == "bottom-left":
        return ["bottom-left", "down", "top-left"]
    if direction == "bottom-right":
        return ["bottom-right", "down", "top-right"]
    return None


def bfs_reachable(board: np.array, start: Tuple[int, int]):
    '''
    Returns the set of reachable points accessible from start, via direct neighbours
    
    # Parameters
    board (numpy array[bool]): Game board with True values at the positions of the player and False elsewhere
    start (Tuple[int, int]): Starting point for the BFS

    # Returns
    Set[Tuple[int, int]]: Set of reachable points accessible from start, via direct neighbours
    '''
    dim = board.shape[0]

    queue = deque([start])
    visited = set()
    visited.add(start)

    while queue:
        current = queue.popleft()
        for nx, ny in get_neighbours(dim, current):
            if is_valid(nx, ny, dim) and (nx, ny) not in visited and board[nx, ny]:
                queue.append((nx, ny))
                visited.add((nx, ny))

    return visited


def find_ring(board: np.array, start: Tuple[int, int]) -> List[Tuple[int, int]]:
    '''
    Returns the points forming a ring with the start point

    # Parameters
    board (numpy array[bool]): Game board with True values at the positions of the player and False elsewhere
    start (Tuple[int, int]): Starting point for the DFS

    # Returns
    List[Tuple[int, int]]: Set of points forming the bridge, via direct neighbours
    '''

    dim = board.shape[0]
    siz = dim // 2  # to determine the half of the board
    directions = ["up", "top-left", "bottom-left", "down"]

    def dfs(board, vertex, direction, visited, path, ring_length):
        if vertex == start and ring_length >= 5:
            return True

        x, y = vertex
        half = np.sign(y - siz)
        new_directions = three_forward_moves(direction)
        for new_dir in new_directions:
            direction_coors = move_coordinates(new_dir, half)
            nx, ny = x + direction_coors[0], y + direction_coors[1]
            if is_valid(nx, ny, dim) and board[nx, ny] and (nx, ny, new_dir) not in visited:
                visited.add((nx, ny, new_dir))
                if dfs(board, (nx, ny), new_dir, visited, path, ring_length + 1):
                    path.append(vertex)
                    return True
                
        return False

    visited = set()
    for direction in directions:
        x, y = start
        half = np.sign(y - siz)  # 0 for mid, -1 for left, 1 for right
        direction_coors = move_coordinates(direction, half)
        nx, ny = x + direction_coors[0], y + direction_coors[1]
        if is_valid(nx, ny, dim) and board[nx, ny] and (nx, ny, direction) not in visited:
            visited.add((nx, ny, direction))
            child_path = [start]
            if dfs(board, (nx, ny), direction, visited, child_path, 0):
                return child_path

    return []


def find_fork(board: np.array, start: Tuple[int, int]) -> List[Tuple[int, int]]:
    '''
    Returns the points forming a fork with the start point
    
    # Parameters
    board (numpy array[bool]): Game board with True values at the positions of the player and False elsewhere
    start (Tuple[int, int]): Starting point for the DFS

    # Returns
    List[Tuple[int, int]]: Set of points forming the bridge, via direct neighbours
    '''
    
    dim = board.shape[0]

    def dfs(board, vertex, visited, path, edges, vis_edge_cnt):
        edge = get_edge(vertex, dim)
        if edge != -1:
            if edge not in edges:
                path.append(vertex)
                edges.append(edge)
            if len(edges) - vis_edge_cnt > 1:
                return

        for nx, ny in get_neighbours(dim, vertex):
            if is_valid(nx, ny, dim) and (nx, ny) not in visited and board[nx, ny]:
                visited.add((nx, ny))
                num_edges = len(edges)
                dfs(board, (nx, ny), visited, path, edges, vis_edge_cnt)
                if len(edges) > num_edges:
                    path.append(vertex)
                if len(edges) - vis_edge_cnt > 1:
                    return

        return

    path = [start]
    visited_edges = []
    edge = get_edge(start, dim)
    if edge != -1:
        visited_edges.append(edge)
    visited = set()
    visited.add(start)
    for nx, ny in get_neighbours(dim, start):
        if is_valid(nx, ny, dim) and board[nx, ny]:
            child_path = []
            num_edges = len(visited_edges)
            dfs(board, (nx, ny), visited, child_path, visited_edges, num_edges)
            if len(visited_edges) > num_edges:
                path.extend(child_path)
            if len(visited_edges) > 2:
                return path

    return path


def find_bridge(board: np.array, start: Tuple[int, int]) -> List[Tuple[int, int]]:
    '''
    Returns the points forming a bridge with the start point
    
    # Parameters
    board (numpy array[bool]): Game board with True values at the positions of the player and False elsewhere
    start (Tuple[int, int]): Starting point for the DFS

    # Returns
    List[Tuple[int, int]]: Set of points forming the bridge, via direct neighbours
    '''
    
    dim = board.shape[0]
    
    def dfs(board, vertex, visited, path):
        corner = get_corner(vertex, dim)
        if corner != -1:
            path.append(vertex)
            return corner

        corner = -1
        for nx, ny in get_neighbours(dim, vertex):
            if is_valid(nx, ny, dim) and (nx, ny) not in visited and board[nx, ny]:
                visited.add((nx, ny))
                corner = dfs(board, (nx, ny), visited, path)
                if corner != -1:
                    path.append(vertex)
                    break

        return corner

    path = [start]
    visited_corner = get_corner(start, dim)
    visited = set()
    visited.add(start)
    for nx, ny in get_neighbours(dim, start):
        if is_valid(nx, ny, dim) and board[nx, ny]:
            child_path = []
            visited.add((nx, ny))
            corner = dfs(board, (nx, ny), visited, child_path)
            if corner != -1 and corner != visited_corner:
                path.extend(child_path)
                if visited_corner == -1:
                    visited_corner = corner
                else:
                    return path

    return path


# Marked (node, incoming direction) visited!
def check_ring(board: np.array, move: Tuple[int, int]) -> bool:
    '''
    Check whether a ring is formed by the move
    
    # Parameters
    board (numpy array[bool]): game board with True values at the positions of the player and False elsewhere
    move (Tuple[int, int]): position of the move. Must have already been played (marked on the board)

    # Returns
    bool: True if a ring is formed by the move, False otherwise
    '''
    # board is already a numpy boolean array, we are only concerned with "true" paths
    # DFS at <move> to check whether ring forms
    dim = board.shape[0]  # of the array
    siz = dim // 2  # to determine the half of the board
    init_move = move
    directions = ["up", "top-left", "bottom-left", "down"]
    visited = set()

    # Trivially false if less than 2 True neighbours present
    neighbours = get_neighbours(dim, move)
    neighbours = [board[neighbour] for neighbour in neighbours]
    if neighbours.count(True) < 2:
        return False

    # In the first step, move in 4 contiguous directions (4 suffices to detect a ring)
    exploration = []
    for direction in directions:
        x, y = move
        half = np.sign(move[1] - siz)  # 0 for mid, -1 for left, 1 for right
        direction_coors = move_coordinates(direction, half)
        nx, ny = x + direction_coors[0], y + direction_coors[1]
        if 0 <= nx < dim and 0 <= ny < dim and board[nx, ny]:
            exploration.append(((nx, ny), direction))
            visited.add((nx, ny, direction))

    ring_length = 1
    # In the later steps, move in 3 "forward" directions (avoids sharp turns)
    while (len(exploration) != 0):
        new_exp = []
        for to_explore in exploration:
            move, prev_direction = to_explore
            x, y = move
            half = np.sign(y - siz)
            new_directions = three_forward_moves(prev_direction)
            for direction in new_directions:
                direction_coors = move_coordinates(direction, half)
                nx, ny = x + direction_coors[0], y + direction_coors[1]
                if is_valid(nx, ny, dim) and board[nx, ny] and (nx, ny, direction) not in visited:
                    if init_move == (nx, ny) and ring_length >= 5:
                        return True
                    new_exp.append(((nx, ny), direction))
                    visited.add((nx, ny, direction))
        exploration = new_exp
        ring_length += 1
    return False


def check_bridge(board: np.array, move: Tuple[int, int]) -> bool:
    '''
    Check whether a bridge is formed by the move, via direct neighbours
    
    # Parameters
    board (numpy array[bool]): game board with True values at the positions of the player and False elsewhere
    move (Tuple[int, int]): position of the move. Must have already been played (marked on the board)

    # Returns
    bool: True if a bridge is formed by the move, False otherwise
    '''
    visited = bfs_reachable(board, move)
    dim     = board.shape[0]
    corners = set(get_all_corners(dim))

    ## check for bridge
    reachable_corners = len([corner for corner in corners if corner in visited])
    if reachable_corners >= 2:
        return True
    return False


def check_fork(board: np.array, move: Tuple[int, int]) -> bool:
    '''
    Check whether a fork is formed by the move, via direct neighbours
    
    # Parameters
    board (numpy array[bool]): game board with True values at the positions of the player and False elsewhere
    move (Tuple[int, int]): position of the move. Must have already been played (marked on the board)

    # Returns
    bool: True if a fork is formed by the move, False otherwise
    '''
    visited = bfs_reachable(board, move)
    dim     = board.shape[0]
    sides   = get_all_edges(dim)
    sides   = [set(side) for side in sides]

    ## check for fork
    reachable_edges = [1 if len(side.intersection(visited)) > 0 else 0 for side in sides]
    if sum(reachable_edges) >= 3:
        return True

    return False


def check_fork_and_bridge(board: np.array, move: Tuple[int, int]) -> Tuple[bool, Union[str, None]]:
    '''
    Check whether a fork or a bridge is formed by the move, via direct neighbours
    
    # Parameters
    board (numpy array[bool]): game board with True values at the positions of the player and False elsewhere
    move (Tuple[int, int]): position of the move. Must have already been played (marked on the board)

    # Returns
    bool: True if a fork or a bridge is formed by the move, False otherwise
    '''
    visited = bfs_reachable(board, move)
    dim     = board.shape[0]
    corners = set(get_all_corners(dim))
    sides = get_all_edges(dim)
    sides = [set(side) for side in sides]

    # check for fork
    reachable_edges = [1 if len(side.intersection(visited)) > 0 else 0 for side in sides]
    if sum(reachable_edges) >= 3:
        # print("Fork Detected!")
        return True, "fork"

    # check for bridge
    reachable_corners = len([corner for corner in corners if corner in visited])
    if reachable_corners >= 2:
        # print("Bridge Detected!")
        return True, "bridge"

    return False, None


def check_win(board: np.array, move: Tuple[int, int], player_num: int, path:List[Tuple[int, int]]=None) -> Tuple[bool, Union[str, None]]:
    '''
    Checks if the player has won the game by placing a move at the given position
    
    # Parameters
    board (numpy array): Game board
    move (Tuple[int, int]): Position of the move. Must have already been played (marked on the board)
    player_num (int): Id of the player who made the move

    # Note
    If the path is not None, any values in the path will be overwritten by the winning path

    # Returns
    bool: True if the player has won the game, False otherwise
    '''
    # Invariant : Win can only be induced through a structure formed at <move> by <player_num>
    # All paths for player_num are set to "True", no other information needed, hence encoded as a lighweight boolean array
    board = (board == player_num)
    if check_ring(board, move):
        if path != None:
            path.clear()
            path.extend(find_ring(board, move))
        return True, "ring"
    
    win, way = check_fork_and_bridge(board, move)
    if win:
        if way == "fork":
            if path != None:
                path.clear()
                path.extend(find_fork(board, move))
        elif way == "bridge":
            if path != None:
                path.clear()
                path.extend(find_bridge(board, move))
        return True, way
    return False, None
