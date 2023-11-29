import csv
import time
from DobutsuBoard import fsf_engine, dobutsu_engine, playself

'''Generate data to train model.'''

default_fsf_depth = 18

moves_fields = ['game_id','turn','move','eval']
gamestats_fields = ['game_id','result','time','depth']


def get_depths(start: int,
                step_size:int,
                  step_count:int):
    '''Generate depth value as a sequence.'''
    yield start  

    if step_count > 0:
        '''
        return get_depths(start + step_size,
                      step_size,
                        step_count - 1)
        '''
        for item in get_depths(start + step_size,
                      step_size,
                        step_count - 1):
            yield item       

game_id = 22

# Remove these lines to generate games using classical eval.
fsf_engine.load_nnue("AIFinalProject\\fairystockfish\\dobutsu-b69e434ed334.nnue")

for depth in get_depths(60, +1, 99):

    dobutsu_engine.reset()
    fsf_engine.reset()

    game_result = playself(fsf_engine,
                                depth,
                                dobutsu_engine)

    start = time.time()

    halfmove_counter = 1

    for record in game_result:
        with open("AIFinalProject\\moves.csv", 'a', newline="\n") as csvfile:
            csvwriter = csv.writer(csvfile)        
            csvwriter.writerow([game_id, halfmove_counter, record.move, record.eval])
            halfmove_counter += 1
    
    end = time.time()
    
    result = dobutsu_engine.state_list[-1].instance.get_result().name

    with open("AIFinalProject\\gamestats.csv", 'a', newline="\n") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([game_id, result, str(end-start), depth])

    game_id += 1