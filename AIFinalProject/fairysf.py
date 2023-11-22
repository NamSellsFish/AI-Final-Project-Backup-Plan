# %%

from typing import Iterable, Protocol
from subprocess import Popen, PIPE

def print_list_vertical(iterable: Iterable, seperator = ""):
    
    current_seperator = ""
    
    for item in iterable:
        print(item)
        print(seperator)
        current_seperator = seperator

def write_process(process: Popen, input: str):

    process.stdin.write((input + '\n').encode()) # type: ignore
    process.stdin.flush() # type: ignore

def readline_process(process: Popen, lines: int):

    result : list[str] = []

    for i in range(lines):
        output = str(process.stdout.readline().decode()[:-1]) # type: ignore
        result.append(output)

    return result

def scroll_process(process: Popen, key: str):

    result : list[str] = []

    while True:
        output = str(process.stdout.readline().decode()[:-1]) # type: ignore
        result.append(output)

        if output.find(key) != -1: return result


class Engine(Protocol):
    def move(self, move: str) -> bool:
        """Returns true if the game is over."""
        ...
    
    def search(self, depth: int) -> str | None: ...
    def reset(self) -> None: ...
    def show_board(self) -> str: ...
    

class FairyStockfishEngine(Engine):
    def __init__(self, path: str) -> None:
        self.__process = Popen([path],
                       stdout=PIPE,
                       stdin=PIPE,
                       stderr=PIPE)
        
        self.moves = ""
        _ = readline_process(self.__process, 1)
        
    def uci(self):
        write_process(self.__process, "uci")
        _ = scroll_process(self.__process, "uciok")

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
                                     "bestmove")[-1][9:13]
    
    def search(self, depth: int) -> str | None:
        result = self.go("depth " + str(depth))

        if result == "(non": return None

        return result

    def reset(self):
        self.moves = ""
        write_process(self.__process,
                      "position startpos")

    def __del__(self):
        write_process(self.__process, "quit")

