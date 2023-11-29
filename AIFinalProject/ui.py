# %%

from typing import Callable, Iterable, TypeVar
import PySimpleGUI as pyGUI
from fairysf import Mate, MoveEvaluation
from DobutsuBoard import DobutsuEngine, DobutsuGameState, ColoredPiece, GameResult, LazilyExpandedTree, PieceDrop, PieceDrops, PieceMove, PieceType, Score, dobutsu_engine, \
    Chick, Elephant, Giraffe, Lion, Hen, flatten, gamestate_to_bits, result_text_map
from sklearn.linear_model import LinearRegression
from joblib import dump, load
# Define the layout

T = TypeVar("T")

import PIL.Image
import io
import base64

#Source: https://stackoverflow.com/a/73295036/21007334
def resize_image(image_path, resize: tuple[int, int] | None):
    '''Resize an image to fit a button.'''
    if isinstance(image_path, str):
        img = PIL.Image.open(image_path)
    else:
        try:
            img = PIL.Image.open(io.BytesIO(base64.b64decode(image_path)))
        except Exception as e:
            data_bytes_io = io.BytesIO(image_path)
            img = PIL.Image.open(data_bytes_io)

    cur_width, cur_height = img.size
    if resize:
        new_width, new_height = resize
        scale = min(new_height/cur_height, new_width/cur_width)
        img = img.resize((int(cur_width*scale), int(cur_height*scale)), PIL.Image.ADAPTIVE)
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    del img
    return bio.getvalue()

img_size = (52, 52)

texture_map = {
    '.' : resize_image('pieces\\empty.png', img_size),

    'C' : resize_image('pieces\\w-pawn.png', img_size),
    'E' : resize_image('pieces\\w-elephant.png', img_size),
    'G' : resize_image('pieces\\w-giraffe.png', img_size),
    'L' : resize_image('pieces\\w-lion.png', img_size),
    'H' : resize_image('pieces\\w-commoner.png', img_size),

    'c' : resize_image('pieces\\b-pawn.png', img_size),
    'e' : resize_image('pieces\\b-elephant.png', img_size),
    'g' : resize_image('pieces\\b-giraffe.png', img_size),
    'l' : resize_image('pieces\\b-lion.png', img_size),
    'h' : resize_image('pieces\\b-commoner.png', img_size),
}

white_hand_color = 'gray'

tile_size = (6, 3)
font = ('Times New Roman', 12)

def get_text(input):
    '''Return Text Element.'''
    return pyGUI.Text(input, size=(5, 1),
                      font=font,
                       justification='center',
                       auto_size_text=True
                       )

white_hand_layout: list[list[pyGUI.Element]] = [
    [
        pyGUI.Button(size=tile_size,
                     pad=(0, 0),
                     button_color=white_hand_color,
                     border_width=3,
                     image_data=texture_map['C'],
                     expand_x=False,
                     expand_y=False,
                     metadata= ColoredPiece(Chick, True),
                     ),

        pyGUI.Button(size=tile_size,
                     pad=(0, 0),
                     button_color=white_hand_color,
                     border_width=3,
                     image_data=texture_map['E'],
                     expand_x=False,
                     expand_y=False,
                     metadata= ColoredPiece(Elephant, True),                     
                     ),

        pyGUI.Button(size=tile_size,
                     pad=(0, 0),
                     button_color=white_hand_color,
                     border_width=3,
                     image_data=texture_map['G'],
                     expand_x=False,
                     expand_y=False,
                     metadata= ColoredPiece(Giraffe, True),
                     )        
    ],
    [
        get_text(0),
        get_text(0),
        get_text(0)
    ]
]

black_hand_layout: list[list[pyGUI.Element]] = [
    [
        pyGUI.Button(size=tile_size,
                     pad=(0, 0),
                     button_color='gray',
                     border_width=3,
                     image_data=texture_map['c'],
                     expand_x=False,
                     expand_y=False,
                     metadata= ColoredPiece(Chick, False),
                     ),

        pyGUI.Button(size=tile_size,
                     pad=(0, 0),
                     button_color='gray',
                     border_width=3,
                     image_data=texture_map['e'],
                     expand_x=False,
                     expand_y=False,
                     metadata= ColoredPiece(Elephant, False),           
                     ),

        pyGUI.Button(size=tile_size,
                     pad=(0, 0),
                     button_color='gray',
                     border_width=3,
                     image_data=texture_map['g'],
                     metadata= ColoredPiece(Giraffe, False),                    
                     )        
    ],
    [
        get_text(0),
        get_text(0),
        get_text(0)
    ]
]

board_layout = [
    [pyGUI.Button(
                    size=tile_size,
                    pad=(0, 0),
                    button_color=(
                        'lightgreen' if y == 0 or y == 3
                        else 'lightyellow'),
                    metadata = (x, 3 - y),
                    border_width=3,
                    expand_x=False,
                    expand_y=False,                 
                )
        for x in range(3)]
            for y in range(4)]

def get_board_tile(input: tuple[int, int]):
    '''
    Use instead of direct indexing,
    since the pos of the matrix of the UI 
    doesn't match that of the internal representation.
    '''
    return board_layout[3 - input[1]][input[0]]

def reload_color():
    '''Reload color of the board.'''
    for button in white_hand_layout[0]:
        button.update(button_color = white_hand_color)
    
    for button in black_hand_layout[0]:
        button.update(button_color = 'gray')

    for x in range(3):
        for y in range(4):
            get_board_tile((x, y)).update(button_color = \
            'lightgreen' if y == 0 or y == 3 \
                        else 'lightyellow')
    
    (board, is_white) = dobutsu_engine.state_list[-1].instance.get_key()

    check_location = board.get_check_source(is_white)

    if check_location == None: return

    get_board_tile(board.get_king_location(is_white)).update(
        button_color='red')

reset_button = pyGUI.Button("Reset")
swap_button = pyGUI.Button("Swap")
status_text = pyGUI.Text()

# Create the window by flattening 3D list to 2D
window = pyGUI.Window('Dobutsu', flatten(
            [
                black_hand_layout,
                board_layout,
                [[status_text]],
                white_hand_layout,
                [[reset_button, swap_button]]
            ],
        ),
    element_justification='center',
    size=(300,500),
    finalize=True
)

previous_element : pyGUI.Element | None = None

dobutsu_engine = DobutsuEngine()
dobutsu_engine.reset()

is_over = False

def load_board(gamestate: DobutsuGameState = None): # type: ignore
    '''Load pieces from a state onto the UI.'''
    global is_over
    if gamestate == None: gamestate = dobutsu_engine.state_list[-1].instance

    ( board, is_white ) = gamestate.get_key()

    symbol_matrix = gamestate.board.to_symbol_array()
    
    for x in range(3):
        for y in range(4):
            symbol = symbol_matrix[x][y]
            
            get_board_tile((x, y)).update(image_data= texture_map[symbol])


    i = 0
    for _, count in gamestate.board.hand[True].items():
        label = white_hand_layout[-1][i]
        
        assert isinstance(label, pyGUI.Text)

        label.update(value= str(count))

        i += 1

    i = 0
    for _, count in gamestate.board.hand[False].items():
        label = black_hand_layout[-1][i]
        
        assert isinstance(label, pyGUI.Text)

        label.update(value= str(count))

        i += 1

    result = board.get_result(is_white)
    if result != GameResult.Ongoing:
        is_over = True
        status_text.update(result_text_map[result])
    else:
        status_text.update(("White" if is_white else "Black") + " to play.")

    reload_color()


def is_hand_tile(element: pyGUI.Element | None):
    '''Check if clicked element is a hand tile.'''
    if element == None: return False
    return isinstance(element.metadata, ColoredPiece)

def is_board_tile(element: pyGUI.Element | None):
    '''Check if clicked element is a board tile.'''
    if element == None: return False
    return isinstance(element.metadata, tuple)

load_board()

selected_color = 'white'
legal_move_color = 'yellow'

def threefold_check():
    '''Check three-fold repition and update status text.'''
    gamestate = dobutsu_engine.state_list[-1].instance

    if gamestate.threefold: 
        status_text.update("Three-fold repitition: 1/2 - 1/2")

def engine_play():
    '''Request the engine to play and output to the board.'''
    
    if is_over: return

    status_text.update("Engine thinking.")

    window.Finalize()
    
    move = dobutsu_engine.search(4)

    if not isinstance(move, MoveEvaluation): return

    dobutsu_engine.move(move.move)

    load_board()

lr : LinearRegression = load('linear_regression.joblin')

def logistics_regression_evaluation(
        node: LazilyExpandedTree[DobutsuGameState]) -> Score:
    return lr.predict([gamestate_to_bits(node.instance)])[0] # type: ignore

dobutsu_engine = DobutsuEngine(logistics_regression_evaluation)


'''
WINDOW LOOP:
'''

engine_turn = False

while True:
    event, values = window.read() # type: ignore
    
    if event == pyGUI.WINDOW_CLOSED:
        break

    if event == None:
        continue

    selected = window[event]

    reload_color()

    if selected == reset_button:
        dobutsu_engine.reset()
        load_board()
        is_over = False
        continue

    if selected == swap_button:
        engine_play()
        continue

    gamestate = dobutsu_engine.state_list[-1].instance

    if gamestate.threefold: 
        status_text.update("Three-fold repitition: 1/2 - 1/2")
        continue

    ( board, is_white ) = dobutsu_engine.state_list[-1].instance.get_key()

    ( legal_moves, legal_drops ) = board.get_legal_moves(is_white) 

    if is_hand_tile(selected):

        colored_piece = selected.metadata

        assert isinstance(colored_piece, ColoredPiece)

        if selected != previous_element and colored_piece.is_white == is_white:

            selected.update(button_color = selected_color)

            locations = [i[0] for i in legal_drops if colored_piece.piece_type in i[1] ]

            for location in locations:
                get_board_tile(location).update(button_color = legal_move_color)

            previous_element = selected

        else:

            previous_element = None

    elif is_board_tile(selected):
        
        location = selected.metadata
        location : tuple[int, int]

        if previous_element == None:
            piece = board.get_piece(location)

            if piece == None or piece.is_white != is_white:
                continue

            selected.update(button_color=selected_color)

            piece_moves = [move for move in legal_moves if move.startpos == location]

            for drop in piece_moves:
                get_board_tile(drop.endpos).update(button_color=legal_move_color)

            previous_element = selected

            threefold_check()
            continue

        elif is_board_tile(previous_element):
            move = PieceMove(previous_element.metadata, location)
            if move in legal_moves:
                dobutsu_engine.move(str(move))
                load_board()

                previous_element = None
                reload_color()
                threefold_check()
                window.Finalize()
                engine_play()
            else:
                previous_element = None

        elif is_hand_tile(previous_element):
            drop = PieceDrop(previous_element.metadata.piece_type, location)
            if len([i for i in legal_drops if i[0] == location and drop.piece_type in i[1]]) == 1:
                dobutsu_engine.move(str(drop))
                load_board()

                previous_element = None
                reload_color()
                threefold_check()
                window.Finalize()
                engine_play()

            else:
                previous_element = None
                     
        else:
            previous_element = None
        
    else:
        previous_element = None
           

window.close()
# %%
