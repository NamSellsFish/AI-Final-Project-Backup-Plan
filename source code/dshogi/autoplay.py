import rule as r

version='0.1'

### Evaluate the board
# XXX: It's primitive
def eval(board):
  if not board: return None
  winner=board.getWinner()
  if winner==r.SENTE:  return 22
  elif winner==r.GOTE: return -22
  else:
    v=0
    for owner in [r.SENTE, r.GOTE]:
      if owner==r.SENTE: sign=1
      else:              sign=-1
      for type in [r.HIYOKO, r.NIWATORI, r.KIRIN, r.ZOU, r.RAION]:
        if type==r.HIYOKO:             d=1
        elif type in [r.ZOU, r.KIRIN]: d=2
        elif type==r.NIWATORI:         d=3
        else:                          d=4
        ks=board.getKomas(owner, type)
        v+=sign*d*len(ks)
    return v

###### Autoplay Class ######
class Autoplay:

  ### Initialize
  def __init__(self, turn, depth=0):
    self.__turn=turn
    self.__depth=depth

  ### Return next action
  # XXX: It's primitive
  def searchAct(self, board, boards, depth=None):
    if not depth: depth=self.__depth
    if not board: return None
    if board.isEnd(boards): return (eval(board), None)
    acts=board.nextActs()
    if not acts: return (eval(board), None)
    if depth<=1:
      pairs=map(lambda x: (eval(board.act(x)), x), acts)
    else:
      es=[]
      for a in acts:
        b=board.act(a)
        boards.append(b)
        es.append(self.searchAct(b, boards, depth-1)[0])
        boards.pop()
      pairs=zip(es, acts)
    if board.getTurn()==r.SENTE: func=max
    else:                        func=min
    return func(pairs)

