# TODO: display customize (default, japanese, simple)
from sys import stdin, stdout
from rule import *
from autoplay import *
from datetime import datetime

###### Construct value ######
HELP='''
HELP:   Print this message.
SHOW:   Print current board.
EXIT:   Exit game.
xy??XY: ??-koma move from xy to XY
        e.g. 'B2HYB3'
??XY:   ??-koma put from mochi-goma to XY
        e.g. 'KRC3'
'''

###### Main procedure
def run():
  # get play mode
  stdout.write('''
Do you want play vs. CPU?
  1. Yes, I want play hard mode.
  2. Yes, I want play nomal mode.
  3. Yes, I want play easy mode.
  4. No, manual mode.
[1-4, default=2]: ''')
  s=stdin.readline()
  if s[0]=='1':   depth=5
  elif s[0]=='3': depth=1
  elif s[0]=='4': depth=None
  else:           depth=3
  if depth:
    stdout.write('''
Sente or Gote?
  1. I play Sente(first move).
  2. I play Gote(passive move).
[1-2, default=1]: ''')
    s=stdin.readline()
    if s[0]=='2': turn=GOTE
    else:         turn=SENTE
    cpu=Autoplay(flipOwner(turn), depth)
  else: cpu=None
  # init
  running=True
  bs=[Board()]
  print bs[-1]
  # main loop
  while running:
    if not cpu or bs[-1].getTurn()==turn:
      # Human's turn
      stdout.write('> ')
      s=stdin.readline()
      s=s.upper()[:-1]
      if s=='HELP':
        print HELP
      elif s=='SHOW':
        print bs[-1]
      elif s=='EXIT':
        running=False
        print 'Bye!'
      elif validAct(s):
        b=bs[-1].act(s)
        if b:
          bs.append(b)
          print b
          if b.isEnd(bs):
            running=False
            winner=b.getWinner()
            if winner: print '%s\'s won!' % (winner)
            else: print 'Draw.'
        else:
          print 'Moving failure! Input again.'
      else:
        print 'What? HELP to print help message.'
    else:
      # CPU's turn
      print 'CPU\'s turn. wait...'
      t=datetime.now()
      pair=cpu.searchAct(bs[-1])
      dt=datetime.now()-t
      print 'CPU takes %d.%d sec.' % (dt.seconds, dt.microseconds/1000)
      print 'CPU action ', pair[1]
      b=bs[-1].act(pair[1])
      bs.append(b)
      print b
  # fin
  return True

if __name__=='__main__':
  run()
