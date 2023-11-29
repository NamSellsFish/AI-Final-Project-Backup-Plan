# %%

from typing import Iterable, Protocol, runtime_checkable
from subprocess import Popen, PIPE

def print_list_vertical(iterable: Iterable, seperator = ""):
    '''Print items in a list vertically.'''
    current_seperator = ""
    
    for item in iterable:
        print(item)
        print(seperator)
        current_seperator = seperator

def write_process(process: Popen, input: str):
    '''Write a line to a process.'''
    process.stdin.write((input + '\n').encode()) # type: ignore
    process.stdin.flush() # type: ignore

def readline_process(process: Popen, lines: int):
    '''Read a number of lines from the output stream of a process.'''

    result : list[str] = []

    for i in range(lines):
        output = str(process.stdout.readline().decode()[:-1]) # type: ignore
        result.append(output)

    return result

def scroll_process(process: Popen, key: str):
    '''
    Read lines from the output stream of a process
    until the key is found.
    '''

    result : list[str] = []

    while True:
        output = str(process.stdout.readline().decode()[:-1]) # type: ignore
        result.append(output)

        if output.find(key) != -1: return result

class Mate:
    '''Define Mate score.'''
    def __init__(self, countdown: int) -> None:
        self.countdown = countdown

    def __str__(self) -> str:
        return "Mate in " + str(self.countdown)

    def __repr__(self) -> str:
        return str(self)

class MoveEvaluation:
    '''Record of move + eval.'''
    def __init__(self, move: str, eval: float | Mate) -> None:
        self.move = move
        self.eval = eval

@runtime_checkable
class Engine(Protocol):
    '''Interface for engine wrappers.'''

    def move(self, move: str) -> bool:
        """Returns true if the game is over."""
        ...
    
    def search(self, depth: int) -> MoveEvaluation | None: ...
    def reset(self) -> None: ...
    def show_board(self) -> str: ...
        
class FairyStockfishEngine(Engine):
    '''
    Wrapper for Fairy Stockfish.
    Uses UCI Protocol to communicate.
    '''

    def __init__(self, path: str) -> None:

        self.__process = Popen([path],
                       stdout=PIPE,
                       stdin=PIPE,
                       stderr=PIPE)
        
        self.moves = ""
        _ = readline_process(self.__process, 1)
        
    def __del__(self):
        self.__process.terminate()
        
    def uci(self):
        write_process(self.__process, "uci")
        _ = scroll_process(self.__process, "uciok")

    def load_nnue(self, path: str):
        write_process(
            self.__process, 
            "setoption name EvalFile value " \
                + path)

    def set_variant(self, variant_name: str):
        write_process(self.__process,
        "setoption name UCI_Variant value " + \
          variant_name)
        
        _ = readline_process(self.__process, 1)

        write_process(self.__process, "position startpos")

    def show_board(self) -> str:
        write_process(self.__process, "d")
        
        lines = scroll_process(self.__process, "Checkers")

        result = ""

        for line in lines:
            result += line + '\n'

        return result
            
    def move(self, move: str) -> bool:
        self.moves += " " + move

        write_process(self.__process, 
                      "position startpos moves" + self.moves)
        
        write_process(self.__process,
                      "go depth 1")
        
        return scroll_process(self.__process, "bestmove")[-1][9:15] == "(none)"
    
    def go(self, args: str):
        write_process(self.__process,
                      "go " + args)
        
        return scroll_process(self.__process,
                                     "bestmove")

    
    def search(self, depth: int) -> MoveEvaluation | None:
        move_stream = self.go("depth " + str(depth))
        move = move_stream[-1][9:13]

        if move_stream[-2].find("mate") != -1:

            eval = Mate(int(move_stream[-2][move_stream[-2].find("mate") + 5 : move_stream[-2].find("nodes") - 1]))

            if eval.countdown == 0: return None

        else:

            eval = float(move_stream[-2][move_stream[-2].find("cp") + 3 : move_stream[-2].find("nodes") - 1])

        return MoveEvaluation(move, eval)

    def reset(self):
        self.moves = ""
        write_process(self.__process,
                      "position startpos")
        
def list_item_concat(moves: list):
    '''Convert list to a string.'''

    result = ""

    for item in moves: result += str(item) + " "

    return result.removesuffix(" ")