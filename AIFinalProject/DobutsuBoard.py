# %%
from enum import Enum
from functools import cmp_to_key
import subprocess
import sys
from types import MappingProxyType
from typing import Callable, Generic, Iterable, Protocol, TypeVar, Union
import numpy as np
from overrides import EnforceOverrides, override
from fairysf import Engine, FairyStockfishEngine, print_list_vertical


'''
y^
|
O-->X (transposed)
'''

class NPArrayWrapper:
    array: np.ndarray

    def __init__(self, array: np.ndarray):
        self.array = array
        
    def __eq__(self, __value: object) -> bool:
        
        if isinstance(__value, NPArrayWrapper):
        
            return np.array_equal(self.array, __value.array)
        
        return False
    
    def __hash__(self) -> int:
        return hash(str(self.array))

def get_empty_3x4_board():
    return NPArrayWrapper(np.full((3, 4), False))

empty_3x4_board = get_empty_3x4_board()


class PieceType:
    
    def __init__(self, name: str, w_symbol: str, b_symbol: str, moves: list[tuple[int, int]]) -> None:
        self.name = name
        self.w_symbol = w_symbol
        self.b_symbol = b_symbol
        self.moves = moves

    def __str__(self) -> str:
        return self.name
    
    def __repr__(self) -> str:
        return self.__str__()


Chick = PieceType('Chick', 'C', 'c', [(0, 1)])
Elephant = PieceType('Elephant', 'E', 'e', [(1, 1), (1, -1), (-1, 1), (-1, -1)])
Giraffe = PieceType('Giraffe', 'G', 'g', [(0, 1), (0, -1), (1, 0), (-1, 0)])
Hen = PieceType('Hen', 'H', 'h', [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1)])
Lion = PieceType('Lion', 'L', 'l', [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)])

all_piecetypes = [ Chick, Elephant, Giraffe, Hen, Lion]
hand_piecetypes = [ Chick, Elephant, Giraffe ]

symbol_dict : dict[str, PieceType] = {}

for piece_type in all_piecetypes:
    symbol_dict[piece_type.w_symbol] = piece_type

def shift_bitboard(bitboard: np.ndarray, vector: tuple[int, int]):
    
    '''
    # Create a new bitboard filled with False
    new_bitboard = np.full(bitboard.shape, False)

    (x, y) = vector

    # Calculate the new indices
    rows, cols = np.indices(bitboard.shape)
    rows = np.clip(rows + x, 0, bitboard.shape[0] - 1)
    cols = np.clip(cols - y, 0, bitboard.shape[1] - 1)

    # Assign the shifted values to the new bitboard
    new_bitboard[rows, cols] = bitboard

    return new_bitboard
    '''
    '''
    new_bitboard = np.roll(bitboard, vector, (0, 1))

    new_bitboard[:vector[0], :] = False
    new_bitboard[:, :vector[1]] = False

    return new_bitboard
    '''

    deq0 = bitboard.tolist()

    result = np.full_like(bitboard, False)
    if vector[0] > 0:
        result[vector[0]:] = bitboard[:-vector[0]]
    elif vector[0] < 0:
        result[:vector[0]] = bitboard[-vector[0]:]
    else:
        result = bitboard.copy()

    bitboard = result.copy()
    result = np.full_like(bitboard, False)
    if vector[1] > 0:
        result[:, vector[1]:] = bitboard[:, :-vector[1]]
    elif vector[1] < 0:
        result[:, :vector[1]] = bitboard[:, -vector[1]:]
    else:
        result = bitboard.copy()

    deq = result.tolist()

    return result

Count = int

def get_locations(bitboard: np.ndarray, cond = True) -> list[tuple[int, int]]:
    locations = np.where(bitboard == cond)
    return list(zip(locations[0], locations[1]))

def add(pos1: tuple[int, int], pos2: tuple[int, int]):
    return (pos1[0] + pos2[0], pos1[1] + pos2[1])

def within_bound(pos: tuple[int, int], bitboard: np.ndarray):
    return 0 <= pos[0] < bitboard.shape[0] \
        and 0 <= pos[1] < bitboard.shape[1]

def to_printable_board(board: np.ndarray):
    return np.flip(board.transpose(), axis=0)

def pos_to_string(pos: tuple[int, int]):
    return chr(pos[0] + 97) + str(pos[1] + 1)

def string_to_pos(input: str):
    return ( ord(input[0]) - 97, int(input[1]) - 1)

class PieceMove:
    def __init__(self,
                 startpos: tuple[int, int],
                 endpos: tuple[int, int]):
        self.startpos = startpos
        self.endpos = endpos

    def __str__(self) -> str:
        return pos_to_string(self.startpos) \
                + pos_to_string(self.endpos)
    
    def __repr__(self) -> str:
        return str(self)
    
    def get_key(self):
        return (self.startpos, self.endpos)

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, PieceMove): return False

        return self.get_key() == __value.get_key()

class PieceDrop:
    def __init__(self,
                 piece_type: PieceType,
                 pos: tuple[int, int]) -> None:
        self.piece_type = piece_type
        self.pos = pos

    def __str__(self) -> str:
        return self.piece_type.w_symbol \
                + "@" \
                + pos_to_string(self.pos)

    def __repr__(self) -> str:
        return str(self)
    
    def get_key(self):
        return (self.piece_type, self.pos)

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, PieceDrop): return False

        return self.get_key() == __value.get_key()

PieceDrops = tuple[tuple[int, int], list[PieceType]]
Move = PieceMove | PieceDrop

class ColoredPiece:
    def __init__(self, piece_type: PieceType, is_white: bool) -> None:
        self.piece_type = piece_type
        self.is_white = is_white

    def get_key(self):
        return (self.piece_type, self.is_white)
    
    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, ColoredPiece): return False

        return self.get_key() == __value.get_key()
        
class DobutsuBoard:
    
    def __init__(self,
                color_boards: dict[bool, np.ndarray],
                piece_boards: dict[PieceType, np.ndarray],
                hand: dict[bool, dict[PieceType, Count]],
                ) -> None:
        self.color_boards = color_boards
        self.piece_boards = piece_boards
        self.hand = hand
        self.key = self.get_key()
    
    def get_king_location(self, is_white: bool):

        board = self.piece_boards[Lion] & self.color_boards[is_white]

        locations = get_locations(board)

        if (len(locations) != 1): raise ValueError("Exactly 1 Lion is permitted. Found: " 
                                                   + str(len(locations)))

        return locations[0]

    def get_defended_squares(self, piece_type: PieceType, is_white: bool):
    
        new_board = np.full((3, 4), False)

        pieces = self.color_boards[is_white] & self.piece_boards[piece_type] 
        
        for vector in piece_type.moves:
            k = 1 if is_white else -1
            target = (k*vector[0], k*vector[1])
            new_board = shift_bitboard(pieces, target) | new_board

        return new_board

    def get_all_defended_squares(self, is_white: bool) -> np.ndarray:

        boards: list[np.ndarray] = []

        for piece_type in self.piece_boards:

            boards.append(self.get_defended_squares(piece_type, is_white))

        result = None

        for board in boards:
            if not isinstance(result, np.ndarray):
                result = board
            else:
                result = result | board

        return result # type: ignore
        

    # Deprecated
    def check_checking(self, is_white: bool):

        king_location = self.get_king_location(is_white)

        for piece_type in self.piece_boards:

            if (piece_type == Lion): continue

            threats = self.get_defended_squares(piece_type, not is_white)
            if (threats[king_location] == True): return True
    
        return False

    def get_check_source(self, is_white: bool):

        return self.get_attack_source(is_white, self.get_king_location(is_white))

    def get_attack_source(self, is_white: bool, pos: tuple[int, int]):

        for direction in Lion.moves:
            target = add(pos, direction)

            if not within_bound(target, self.color_boards[True]): continue

            if self.color_boards[not is_white][target]:

                piece_type = self.get_piece(target).piece_type # type: ignore

                for move in piece_type.moves:

                    k = -1 if is_white else 1
                    if move == (k*direction[0], k*direction[1]): return target

    def get_piece(self, pos: tuple[int, int]) -> ColoredPiece | None:

        is_none = True

        color = False

        for color in [True, False]:
            if self.color_boards[color][pos] == True:
                is_none = False
                break

        if is_none: return None

        for piece_type, bitboard in self.piece_boards.items():
            if bitboard[pos] == True:
                return ColoredPiece(piece_type, color)
            
        return None

    def add_to_hand_dict(self, piece_type: PieceType, is_white: bool):
        
        result = value_copy_dict(self.hand)

        if piece_type == Hen: piece_type = Chick

        result[is_white][piece_type] += 1

        return result

    def is_king_win(self, is_white: bool):
        location = self.get_king_location(is_white)
        return location[1] == (3 if is_white else 0)

    def move_piece(self, startpos: tuple[int, int], endpos: tuple[int, int]):

        color_dict = value_copy_dict(self.color_boards)
        piece_dict = value_copy_dict(self.piece_boards)
        hand_dict = value_copy_dict(self.hand)

        is_none = True

        moveside_color = False

        for color in [True, False]:
            if color_dict[color][startpos] == True:
                color_dict[color][startpos] = False
                color_dict[color][endpos] = True
                is_none = False
                moveside_color = color
            else:
                color_dict[color][endpos] = False
                

        if is_none: raise ValueError("Cannot move empty square: " 
                                     + str(startpos) + " to "
                                     + str(endpos))

        moved_piece = None
        captured_piece = None

        for piece_type, bitboard in piece_dict.items():
            if bitboard[startpos] == True:
                moved_piece = piece_type
            if bitboard[endpos] == True:
                captured_piece = piece_type
                bitboard[endpos] = False

            bitboard[startpos] = False            
    
            if moved_piece != None and captured_piece != None: break 

        if moved_piece == Chick \
            and endpos[1] == (3 if moveside_color else 0):
            moved_piece = Hen

        piece_dict[moved_piece][endpos] = True # type: ignore

        if captured_piece != None:
            # warning: functions should check if enemy king is checked.
            if captured_piece == Lion:
                return self
            hand_dict = self.add_to_hand_dict(captured_piece, moveside_color)

        return DobutsuBoard(color_dict, piece_dict, hand_dict)

    def drop_piece(self, piece_type: PieceType, pos: tuple[int, int], is_white: bool):

        if (self.color_boards[True] & self.color_boards[False])[pos]: raise ValueError("Cannot drop on occupied square.")
        if self.hand[is_white][piece_type] <= 0: raise ValueError("Insufficient Piece Count to drop.")

        color_dict = value_copy_dict(self.color_boards)
        piece_dict = value_copy_dict(self.piece_boards)
        hand_dict = value_copy_dict(self.hand)

        color_dict[is_white][pos] = True
        piece_dict[piece_type][pos] = True
        hand_dict[is_white][piece_type] -= 1

        return DobutsuBoard(color_dict, piece_dict, hand_dict)

    def move(self, move: Move, is_white: bool):
        if isinstance(move, PieceMove):
            return self.move_piece(move.startpos,
                                    move.endpos)
        elif isinstance(move, PieceDrop):
            return self.drop_piece(move.piece_type,
                                    move.pos,
                                      is_white)
        
    def get_mobility(self, is_white: bool) -> float:
        result = 0

        empty_squares = get_locations(self.color_boards[True] | self.color_boards[False], False)

        for _, count in self.hand[is_white].items():
            result += len(empty_squares) * (1 if count > 0 else 0)

        for piece_type, board in self.piece_boards.items():
            result += len(
                get_locations(
                self.get_defended_squares(
                    piece_type, is_white)))

        return result

    def get_legal_moves(self, is_white: bool) -> tuple[list[PieceMove], list[PieceDrops]]:
        
        if self.get_check_source(is_white) != None:

            return self._get_legal_moves_if_checked(is_white)

        return self._get_legal_moves_if_no_checks(is_white)
    
    def _get_legal_moves_if_checked(self, is_white: bool):

        moves : list[PieceMove] = []

        drops : list[PieceDrops] = []

        for direction in Lion.moves:

            source = self.get_check_source(is_white)

            target = add(source, direction) # type: ignore

            if not within_bound(target, empty_3x4_board.array): continue

            if self.color_boards[not is_white][target]:

                piece_type = self.get_piece(target).piece_type # type: ignore

                if piece_type == Lion: continue

                for move in piece_type.moves:

                    k = -1 if is_white else 1
                    if move == (k*target[0], k*target[1]): moves.append(PieceMove(target, source)) # type: ignore

            king_location = self.get_king_location(is_white)

            escape = add(king_location, direction)

            if not within_bound(escape, empty_3x4_board.array) \
                or self.color_boards[is_white][escape]: continue

            attacked = self.get_all_defended_squares(not is_white)

            if not attacked[escape]:
                moves.append(PieceMove(king_location, escape))

        return ( moves, drops )
    
    def _get_legal_moves_if_no_checks(self, is_white):

        moves : list[PieceMove] = []

        drops : list[PieceDrops] = []

        empty_squares = get_locations(self.color_boards[True] | self.color_boards[False], False)

        for square in empty_squares:

            dropable: list[PieceType] = []

            for piece_type, count in self.hand[is_white].items():
                if count > 0: dropable.append(piece_type)

            if len(dropable) > 0: drops.append((square, dropable))
                
        piece_locations = get_locations(self.color_boards[is_white], True)

        for location in piece_locations:
            piece_type = self.get_piece(location).piece_type # type: ignore

            for move in piece_type.moves:

                k = 1 if is_white else -1
                target = add(location, (k*move[0],k*move[1]))

                if piece_type == Lion \
                    and self.get_attack_source(is_white, target) != None:
                    continue

                if within_bound(target, empty_3x4_board.array) \
                    and not self.color_boards[is_white][target] \
                    and not self.piece_boards[Lion][target]:
                    moves.append(PieceMove(location, target))

        return (moves, drops)

    def to_symbol_array(self) -> np.ndarray[str, str]: # type: ignore
        result = np.full((3, 4), "")

        for x in range(0, 3):
            for y in range(0, 4):

                piece_obj = self.get_piece((x, y)) 

                if piece_obj == None:
                    result[(x, y)] = "."
                else:
                    result[(x, y)] = \
                        piece_obj.piece_type.w_symbol \
                        if piece_obj.is_white \
                            else piece_obj.piece_type.b_symbol

        return result # type: ignore
    
    def count(self, piece_type: PieceType, is_white: bool):
        return len(get_locations(self.piece_boards[piece_type] & self.color_boards[is_white]))

    def print(self):
        
        print(to_printable_board(self.to_symbol_array()))

        print(self.hand)

    def get_key(self):

        items = []

        for color in [True, False]:
            items.append(NPArrayWrapper(self.color_boards[color]))
            
            for piece_type in hand_piecetypes:
                items.append(self.hand[color][piece_type])  
            
        for piece_type in all_piecetypes:
            items.append(NPArrayWrapper(self.piece_boards[piece_type]))

        return tuple(items)

    def get_result(self, is_white: bool):

        in_check = self.get_check_source(is_white) == None

        moves = self.get_legal_moves(is_white)

        over = len(moves[0]) == 0 and len(moves[1]) == 0

        if not over: return GameResult.Ongoing

        if over and not in_check: return GameResult.Draw
        if over and is_white: return GameResult.BlackWin

        return GameResult.WhiteWin

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, DobutsuBoard): return False

        return self.key == __value.key
    
    def __hash__(self) -> int:
        return hash(self.key)

def value_copy_dict(dictionary: dict):
    result = {}

    for key, value in dictionary.items():
        result[key] = value.copy()

    return result

black_board = np.array([
    [0, 0, 0, 1],
    [0, 0, 1, 1],
    [0, 0, 0, 1],
    ]).astype(bool)

white_board = np.array([
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [1, 0, 0, 0],
    ]).astype(bool)

color_boards = { True: white_board,
                 False: black_board }

chick_board = np.array([
    [0, 0, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 0],
    ]).astype(bool)

elephant_board = np.array([
    [1, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    ]).astype(bool)

giraffe_board = np.array([
    [0, 0, 0, 1],
    [0, 0, 0, 0],
    [1, 0, 0, 0],
    ]).astype(bool)

lion_board = np.array([
    [0, 0, 0, 0],
    [1, 0, 0, 1],
    [0, 0, 0, 0],
    ]).astype(bool)

hen_board = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    ]).astype(bool)

piece_boards = {
    Chick: chick_board,
    Elephant: elephant_board,
    Giraffe: giraffe_board,
    Lion: lion_board,
    Hen: hen_board
}

hands = {
    True : 
    {
        Chick : 0,
        Elephant : 0,
        Giraffe : 0
    },
    False :
    {
        Chick : 0,
        Elephant : 0,
        Giraffe : 0
    }
}

class GameResult(Enum):
    Ongoing = 0
    WhiteWin = 1
    BlackWin = 2
    Draw = 3

T = TypeVar("T")
TAction = TypeVar("TAction")
TNode = TypeVar("TNode", covariant=True, bound="Node")

class Node(Generic[TNode], EnforceOverrides):
    ''' 
    Provides a "view" of a tree-like data structure.
    '''

    def children(self) -> list[TNode]:
        raise NotImplementedError("Please Implement this method")
    
    def parent(self) -> TNode | None:
        return None
    
ChildrenProducer = Callable[[T], list[T]]

class LazilyExpandedTree(Generic[T], Node["LazilyExpandedTree[T]"]):
    '''
    Wraps around an instance that does not know its children.
    Lazily compute its children when children() is called.
    '''

    instance: T
    _compute_children: ChildrenProducer["LazilyExpandedTree[T]"]
    _matrix_init = False
    _children: list["LazilyExpandedTree[T]"] = []
    _parent: Union["LazilyExpandedTree[T]", None]

    def __init__(self, instance: T,
                compute_children: 
                ChildrenProducer["LazilyExpandedTree[T]"],
                parent: Union["LazilyExpandedTree[T]", None] = None) -> None:
        super().__init__()

        self.instance = instance
        self._compute_children = compute_children
        self._parent = parent

    @override
    def parent(self):
        return self._parent

    @override
    def children(self) -> list["LazilyExpandedTree[T]"]:
        '''Children are computed once this property is called, and only once.'''
        if self._matrix_init == False:
            self._children = self._compute_children(self)
            self._matrix_init = True
        return self._children

class DobutsuGameState:

    threefold = False

    def __init__(self, board: DobutsuBoard,
                 is_white: bool,
                 appearance_count: MappingProxyType[tuple[DobutsuBoard, bool], Count],
                 previous_move: Move | None):
        self.board = board
        self.is_white = is_white
        self.previous_move = previous_move

        if (board, is_white) in appearance_count:

            temp = dict(appearance_count)

            temp[(board, is_white)] += 1

            self.appearance_count = MappingProxyType(temp)
        else:
            self.appearance_count = MappingProxyType(appearance_count 
                                                     | {(board, is_white): 1 })
            
        if self.appearance_count[(board, is_white)] > 2: 
            self.threefold = True

    def get_key(self):
        return (self.board, self.is_white)
    
    def move(self, move: Move):
        return DobutsuGameState(
            self.board.move(move, self.is_white),
            not self.is_white,
            self.appearance_count,
            move
        )
    
    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, DobutsuGameState): return False

        return self.get_key() == __value.get_key()
    
    def __hash__(self) -> int:
        return hash(self.get_key())


class StateAction(Generic[T, TAction]):
    def __init__(self, state: T, previous_action: TAction | None):
        self.state = state
        self.previous_action = previous_action
    
    state: T
    previous_action: TAction | None

LazySolutionTree = LazilyExpandedTree[StateAction[T, TAction]]

Score = float
class Mate:
    def __init__(self, countdown: int) -> None:
        self.countdown = countdown

    def __str__(self) -> str:
        return "Mate in " + str(self.countdown)

    def __repr__(self) -> str:
        return str(self)
    
def generate_children(node: LazilyExpandedTree[DobutsuGameState]):
    
    if node.instance.threefold: return list[LazilyExpandedTree[DobutsuGameState]]()
    
    children_states : list[DobutsuGameState] = []

    is_white = node.instance.is_white
    board = node.instance.board

    if board.is_king_win(is_white): return []

    actions = board.get_legal_moves(is_white)

    for move in actions[0]:
        children_states.append(node.instance.move(move))

    for drops in actions[1]:
        for piece_type in drops[1]:
            children_states.append(
                node.instance.move(
                    PieceDrop(piece_type, drops[0])
                ))

    return [LazilyExpandedTree
            (
                i,
                generate_children,
                node
            )
             for i in children_states]

def alternate(val: Score | Mate):
    if isinstance(val, Mate):
        if val.countdown <= 0: return Mate(-(val.countdown - 1))
        return Mate(-val.countdown)
    return -val

def get_optimal(val1: Score | Mate, val2: Score | Mate):

    if isinstance(val1, Mate) and isinstance(val2, Mate):
        if val1.countdown == 0 or val2.countdown == 0:
            return Mate(0)
        if val1.countdown * val2.countdown > 0:
            return Mate(min(val1.countdown, val2.countdown))   
        return Mate(max(val1.countdown, val2.countdown))
        
    elif isinstance(val1, float) and isinstance(val2, float):
        return max(val1, val2)
    elif isinstance(val1, float) and isinstance(val2, Mate):
        if val2.countdown < 0: return val1
        return val2
    if val1.countdown < 0: return val2 # type: ignore
    return val1

def compare_score(val1: Score | Mate, val2: Score | Mate):

    if isinstance(val1, Mate) and isinstance(val2, Mate):

        if val1.countdown * val2.countdown >= 0:
            return val2.countdown - val1.countdown
        elif val1.countdown > 0: return 1
        return -1
    
    elif isinstance(val1, float) and isinstance(val2, float):
        return int(val1 - val2)
    
    elif isinstance(val1, float) and isinstance(val2, Mate):
        if val2.countdown < 0: return 1
        return -1
    if val1.countdown < 0: return -1 # type: ignore
    return 1

def negamax(node: LazilyExpandedTree[T], depth: int,
             func: Callable[[LazilyExpandedTree[T]], Score | Mate]) -> Score | Mate:
    if depth == 0 or len(node.children()) == 0: return func(node)

    value = Mate(-sys.maxsize)

    for child in node.children():
        
        value = get_optimal(value,
                             alternate(
                                 negamax(child,
                                          depth - 1,
                                            func)))

    return value

def negamax_moves(node: LazilyExpandedTree[DobutsuGameState], depth: int,
             func: Callable[[LazilyExpandedTree[DobutsuGameState]], Score | Mate]):
    assert depth > 0

    (board, is_white) = node.instance.get_key()

    moves: list[tuple[Move, Score | Mate]]  = []

    for child in node.children():
        moves.append((
            child.instance.previous_move, # type: ignore
            alternate(negamax(child, depth - 1, func))
        ))

    return sorted(moves, key=cmp_to_key(
    lambda x, y: -compare_score(x[1], y[1])))

inhand_multiplier = 1

# more moves + king_advancement = better
def heuristic_sample1(node: LazilyExpandedTree[DobutsuGameState]) -> Score | Mate:
    
    (board, is_white) = node.instance.get_key()

    enemy_node = node if node.instance.is_white else node.parent() 

    actions = board.get_legal_moves(is_white)

    if len(actions[0]) == 0:
        if board.get_check_source(is_white) != None \
            or board.is_king_win(is_white):
            return Mate(0)
        if len(actions[1]) == 0:
            return 0.0

    return (heuristic_eval_sample1(node)
            - heuristic_eval_sample1(enemy_node, # type: ignore
                not node.instance.is_white)) \
                * (1 if node.instance.is_white else -1)

def heuristic_eval_sample1(node: LazilyExpandedTree[DobutsuGameState], is_white: bool = None) -> Score | Mate: # type: ignore
    
    board = node.instance.board
    if is_white == None: is_white = node.instance.is_white
       
    king_location = board.get_king_location(is_white)

    king_advance = king_location[1] if is_white else 3 - king_location[1]

    return board.get_mobility(is_white) + 1.1* king_advance

def input_to_move(input: str) -> Move:
    if len(input) != 4: 
        raise ValueError("Length was " + str(len(input)))

    endpos = string_to_pos(input[2:4])

    if input[1] == "@":
        dropped_piece = symbol_dict[input[0]]
        

        return PieceDrop(dropped_piece, endpos)
    
    startpos = string_to_pos(input[0:2])

    return PieceMove(startpos, endpos)

board = DobutsuBoard(color_boards, piece_boards, hands)

root_state = LazilyExpandedTree \
(
    DobutsuGameState(board,
                      True,
                        MappingProxyType({}),
                          None),
    generate_children
)

def get_root_state_list():
    state_list : list[LazilyExpandedTree[DobutsuGameState]] = []
    state_list.append(root_state)

    return state_list

class DobutsuEngine(Engine):

    state_list: list[LazilyExpandedTree[DobutsuGameState]] = []

    def __init__(self) -> None:
        self.reset()

    def move(self, move: str) -> bool:

        self.state_list.append(
            LazilyExpandedTree \
        (
            self.state_list[-1].instance.move(
                input_to_move(move)),
            generate_children,
            self.state_list[-1]
        ))

        if (len(self.state_list[-1].children()) == 0): 
            return True
        
        return False
              
    def search(self, depth: int) -> str | None:
        evaluation = negamax_moves(self.state_list[-1],
                    depth,
                    heuristic_sample1)
        
        return str(evaluation[0][0]) if len(evaluation) > 0 \
                else None

    def reset(self) -> None:
        self.state_list = get_root_state_list()

    def show_board(self) -> str:

        board = self.state_list[-1].instance.board

        rows = to_printable_board(board.to_symbol_array())
        result = ""

        for row in rows:
            result += str(row) + '\n'

        result += str(board.hand)

        return result

class PlayerEngine(Engine):
    def __init__(self, engine: Engine):
        self.__engine = engine

    def move(self, move: str) -> bool:
        return self.__engine.move(move)
    
    def search(self, depth: int) -> str | None:
        return input()

    def reset(self) -> None:
        self.__engine.reset()
    def show_board(self) -> str:
        return self.__engine.show_board()
        

class Gameplay:

    def __init__(self, movelist: list[str], gamestates: list[str]) -> None:
        self.movelist = movelist
        self.gamestates = gamestates
    
    def combine(self):
        result: list[str] = []

        for i in range(len(self.movelist)):
            result.append(f"[{i}]: " + self.movelist[i])
            result.append(self.gamestates[i])

        return result

def play(white_engine: Engine, black_engine: Engine,
          white_depth: int, black_depth: int) -> Gameplay:
    
    movelist : list[str] = []
    gamestates: list[str] = []

    engines = [white_engine, black_engine]
    depths = [white_depth, black_depth]

    while(True):
        for i in range(2):
            move = engines[i].search(depths[i])

            if (move == None):
                return Gameplay(movelist,
                                 gamestates)

            movelist.append(move)

            for k in range(2):
                _ = engines[k].move(move)

            gamestates.append(engines[i].show_board())

print("Begin:")

dobutsu_engine = DobutsuEngine()

fsf_engine = FairyStockfishEngine("fairy-stockfish_x86-64.exe")
fsf_engine.uci()
fsf_engine.set_variant('dobutsu')

# result = play(dobutsu_engine, fsf_engine, 1, 6)
# print_list_vertical(result.combine(), "===================================") 
# %%