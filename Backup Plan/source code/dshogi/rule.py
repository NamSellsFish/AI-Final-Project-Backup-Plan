import re

####### Constrait value #######

version='0.1'

# koma's type
HIYOKO  ='HY'
NIWATORI='NW'
ZOU     ='ZO'
KIRIN   ='KR'
RAION   ='RI'

# owner
SENTE  ='SENTE'
GOTE   ='GOTE'

####### Functions #######
### Check owner
def validOwner(owner):
  if owner.upper() in [SENTE, GOTE]:
    return True
  else:
    return False

### Check type
def validType(type):
  if type.upper() in [HIYOKO, NIWATORI, KIRIN, ZOU, RAION]:
    return True
  else:
    return False

### Check pos
def validPos(pos):
  r=re.compile('[A-C][1-4]|MOCHI')
  if r.match(pos.upper()): return True
  else:                    return False

### Check koma
def validKoma(koma):
  return validOwner(koma.getOwner()) and validType(koma.getType()) and validPos(koma.getPos())

### Check act
def validAct(act):
  r=re.compile('([A-C][1-4])?(HY|NW|ZO|KR|RI)([A-C][1-4])(\\s*NARAZU)?')
  if r.match(act.upper()):  return True
  else:                     return False

### Flip owner
def flipOwner(owner):
  if owner.upper()==SENTE: return GOTE
  else:                    return SENTE

### Return moved pos
###   e.g. movedPos('A1', GOTE, 'ForwardRight') returned 'B2'
def movedPos(pos, owner, dir):
  post=''
  if dir.find('Left')!=-1:
    if pos[0]=='B':   post+='A'
    elif pos[0]=='C': post+='B'
    else:             post+='?'
  elif dir.find('Right')!=-1:
    if pos[0]=='A':   post+='B'
    elif pos[0]=='B': post+='C'
    else:             post+='?'
  else:               post+=pos[0]
  if owner==SENTE: fwd=-1
  else:            fwd=1
  if dir.find('Forward')!=-1: post+=str(int(pos[1])+fwd)
  elif dir.find('Back')!=-1:  post+=str(int(pos[1])-fwd)
  else:                       post+=pos[1]
  return post

###### Koma class ######
### properties:
###   pos: koma position
###        (e.g. 'A2', 'C4', 'MOCHI')
###
###   type: koma type
###         (e.g. HIYOKO, RAION)
###
###   owner: koma's owner
###          (e.g. 'SENTE', 'GOTE')
class Koma:

  ###### Instance Method ######
  ### Initialize
  def __init__(self, owner, type, pos):
    # check
    if not validOwner(owner): raise ValueError('Invalid owner is setting.')
    if not validType(type):  raise ValueError('Invalid owner is setting.')
    if not validPos(pos):     raise ValueError('Invalid pos is setting.')
    self.__owner=owner
    self.__type=type
    self.__pos=pos

  ### To string
  def __str__(self):
    s='%s\'s %s on %s' % (self.getOwner(), self.getType(), self.getPos())
    return s

  ### '==' operator
  def __eq__(self, other):
    if not other:                           return False
    elif self.getOwner()!=other.getOwner(): return False
    elif self.getType()!=other.getType():   return False
    elif self.getPos()!=other.getPos():     return False
    return True

  ### '!=' operator
  def __ne__(self, other):
    return not self==other

  ### '<' operator
  def __lt__(self, other):
    if self.getOwner()<other.getOwner(): return True
    if self.getType()<other.getType():   return True
    if self.getPos()<other.getPos():     return True

  ### '<=' operator
  def __le__(self, other):
    return self<other or self==other

  ### '>' operator
  def __gt__(self, other):
    if self.getOwner()>other.getOwner(): return True
    if self.getType()>other.getType():   return True
    if self.getPos()>other.getPos():     return True

  ### '>=' operator
  def __ge__(self, other):
    return self>other or self==other

  ### compare
  def __cmp__(self, other):
    if self==other: return 0
    if self>other:  return 1
    if self<other:  return -1

  ### hash value
  def __hash__(self):
    v=0
    v^=hash(self.__owner)
    v^=hash(self.__type)
    v^=hash(self.__pos)
    return v

  ### Gets
  def getOwner(self): return self.__owner
  def getType(self):  return self.__type
  def getPos(self):   return self.__pos

  ### Check the koma is mochi-koma
  def isMochigoma(self):
    return self.getPos()=='MOCHI'

  ### Get movable positions list
  def getMovables(self):
    if self.isMochigoma():
      all=[]
      for x in ['A', 'B', 'C']:
        for y in ['1', '2', '3', '4']:
          all.append(x+y)
      return all
    owner=self.getOwner()
    pos=self.getPos()
    movables=[]
    if self.getType()==HIYOKO:
      movables.append(movedPos(pos, owner, 'Forward'))
    elif self.getType()==NIWATORI:
      movables.append(movedPos(pos, owner, 'Forward'))
      movables.append(movedPos(pos, owner, 'Back'))
      movables.append(movedPos(pos, owner, 'Left'))
      movables.append(movedPos(pos, owner, 'Right'))
      movables.append(movedPos(pos, owner, 'ForwardRight'))
      movables.append(movedPos(pos, owner, 'ForwardLeft'))
    elif self.getType()==ZOU:
      movables.append(movedPos(pos, owner, 'ForwardRight'))
      movables.append(movedPos(pos, owner, 'ForwardLeft'))
      movables.append(movedPos(pos, owner, 'BackRight'))
      movables.append(movedPos(pos, owner, 'BackLeft'))
    elif self.getType()==KIRIN:
      movables.append(movedPos(pos, owner, 'Forward'))
      movables.append(movedPos(pos, owner, 'Back'))
      movables.append(movedPos(pos, owner, 'Left'))
      movables.append(movedPos(pos, owner, 'Right'))
    elif self.getType()==RAION:
      movables.append(movedPos(pos, owner, 'Forward'))
      movables.append(movedPos(pos, owner, 'Back'))
      movables.append(movedPos(pos, owner, 'Left'))
      movables.append(movedPos(pos, owner, 'Right'))
      movables.append(movedPos(pos, owner, 'ForwardRight'))
      movables.append(movedPos(pos, owner, 'ForwardLeft'))
      movables.append(movedPos(pos, owner, 'BackRight'))
      movables.append(movedPos(pos, owner, 'BackLeft'))
    return movables

  ### Check the koma can move the position
  def canMove(self, pos):
    if not validPos(pos): return False
    if not self.isMochigoma():
      if not (pos.upper() in self.getMovables()): return False
    return True

###### Board class ######
### properties:
###   turn: next actor
###   ks: koma list
class Board():
  ###### Instance Method ######
  ### Initialize
  def __init__(self, komas=None, turn=None):
    if not komas and not turn:
      self.__ks=[]
      self.__ks.append(Koma(SENTE, KIRIN,  'C4'))
      self.__ks.append(Koma(SENTE, RAION,  'B4'))
      self.__ks.append(Koma(SENTE, ZOU,    'A4'))
      self.__ks.append(Koma(SENTE, HIYOKO, 'B3'))
      self.__ks.append(Koma(GOTE,  KIRIN,  'A1'))
      self.__ks.append(Koma(GOTE,  RAION,  'B1'))
      self.__ks.append(Koma(GOTE,  ZOU,    'C1'))
      self.__ks.append(Koma(GOTE,  HIYOKO, 'B2'))
      self.__turn=SENTE
    elif not komas or not turn:
      raise ValueError('Koma list and turn are all or nothing')
    else:
      for k in komas:
        if not validKoma(k):
          raise ValueError('Invalid koma list is setting')
      if not validOwner(turn):
        raise ValueError('Invalid next actor is setting')
      self.__ks=komas
      self.__turn=turn
    self.__ks.sort()

  ### To string
  def __str__(self):
    s=''
    ms=[m.getType().upper() for m in self.__ks if m.getOwner()==GOTE and m.getPos()=='MOCHI']
    s+='GOTE MOCHI-GOMA: %s' % (ms)
    s+='\n'
    s+=' A  B  C\n'
    for y in ['1', '2', '3', '4']:
      for x in ['A', 'B', 'C']:
        k=self.getKoma(x+y)
        if k==None:               s+='-- '
        elif k.getOwner()==SENTE: s+=k.getType().lower()+' '
        elif k.getOwner()==GOTE:  s+=k.getType().upper()+' '
        else:                     s+='?? '
      s+=' %s\n' % (y)
    ms=[m.getType().lower() for m in self.__ks if m.getOwner()==SENTE and m.getPos()=='MOCHI']
    s+='sente mochi-goma: %s\n' % (ms)
    s+='%s\'s turn' % (self.__turn)
    return s

  ### '==' operator
  def __eq__(self, other):
    if not other:                         return False
    elif self.__ks!=other.__ks:           return False
    elif self.getTurn()!=other.getTurn(): return False
    return True

  ### '!=' operator
  def __ne__(self, other):
    return not self==other

  ### '<' operator
  def __lt__(self, other):
    if self.__ks<other.__ks:           return True
    if self.getTurn()<other.getTurn(): return True

  ### '<=' operator
  def __le__(self, other):
    return self<other or self==other

  ### '>' operator
  def __gt__(self, other):
    if self.__ks>other.__ks:           return True
    if self.getTurn()>other.getTurn(): return True

  ### '>=' operator
  def __ge__(self, other):
    return self>other or self==other

  ### compare
  def __cmp__(self, other):
    if self==other: return 0
    if self>other:  return 1
    if self<other:  return -1

  ### Hash value
  def __hash__(self):
    v=0
    for k in self.__ks: v=v^hash(k)
    if self.__turn:     v<<=1
    return v

  ### Get the koma with position
  def getKoma(self, pos):
    ks=[k for k in self.__ks if k.getPos()==pos]
    if len(ks)==0: return None
    else:          return ks[0]

  ### Get the koma with owner and type
  def getKomas(self, owner, type):
    ks=[k for k in self.__ks if k.getOwner()==owner and k.getType()==type]
    return ks

  ### Get the mochi-goma with owner and type
  def getMochigoma(self, owner, type):
    ks=[k for k in self.__ks if k.getOwner()==owner and k.getType()==type and k.getPos()=='MOCHI']
    if len(ks)==0: return None
    else:          return ks[0]

  ### Get the mochi-goma list with owner
  def getMochigomas(self, owner):
    ks=[k for k in self.__ks if k.getOwner()==owner and k.getPos()=='MOCHI']
    return ks

  ### Get turn
  def getTurn(self):
    return self.__turn

  ### Check action
  def canMove(self, act):
    r=re.compile('([A-C][1-4])?(HY|NW|ZO|KR|RI)([A-C][1-4])(\\s*NARAZU)?')
    m=r.match(act.upper())
    if not m: return False
    pre =m.group(1)
    type=m.group(2)
    post=m.group(3)
    narazu=m.group(4)
    # check
    if pre:
      k=self.getKoma(pre)
      if not k:                        return False
      if k.getOwner()!=self.getTurn(): return False
      if k.getType()!=type:            return False
      if not k.canMove(post):          return False
      kk=self.getKoma(post)
      if kk and kk.getOwner()==self.getTurn(): return False
    else:
      k=self.getMochigoma(self.getTurn(), type)
      if not k: return False
      kk=self.getKoma(post)
      if kk: return False
    return True

  ### Action and return new board
  def act(self, act):
    if not self.canMove(act): return None
    r=re.compile('([A-C][1-4])?(HY|NW|ZO|KR|RI)([A-C][1-4])(\\s*NARAZU)?')
    m=r.match(act.upper())
    pre =m.group(1)
    type=m.group(2)
    post=m.group(3)
    narazu=m.group(4)
    if pre:
      k=self.getKoma(pre)
      kk=self.getKoma(post)
    else:
      k=self.getMochigoma(self.getTurn(), type)
      kk=self.getKoma(post)
    newks=[]
    for oldk in self.__ks:
      newk=Koma(oldk.getOwner(), oldk.getType(), oldk.getPos())
      newks.append(newk)
    if pre and k.getType()==HIYOKO and (post[1]=='1' or post[1]=='4') and not narazu:
        type=NIWATORI
    newks.remove(k)
    newks.append(Koma(self.getTurn(), type, post))
    if kk:
      if kk.getType()==NIWATORI: type=HIYOKO
      else:                      type=kk.getType()
      newks.remove(kk)
      newks.append(Koma(self.getTurn(), type, 'MOCHI'))
    b=Board(newks, flipOwner(self.getTurn()))
    return b

  ### Get the game winner
  ###   if the game is not end, return None
  def getWinner(self):
    ks=[k for k in self.__ks if k.getType()==RAION]
    r1=re.compile('.1')
    r4=re.compile('.4')
    for k in ks:
      if k.getPos()=='MOCHI': return k.getOwner()
      if k.getOwner()==SENTE and r1.match(k.getPos()): return SENTE
      if k.getOwner()==GOTE  and r4.match(k.getPos()): return GOTE
    return None

  ### Check the game is end
  def isEnd(self, boards):
    if self.getWinner()!=None or boards.count(self)>=3: return True 
    else: return False

  ### Get next action list of the board
  def nextActs(self):
    #if self.isEnd(): return []
    acts=[]
    for k in filter(lambda x: x.getOwner()==self.getTurn(), self.__ks):
      movs=k.getMovables()
      for mov in movs:
        act=k.getPos()+k.getType()+mov
        if self.canMove(act): acts.append(act)
    return acts

  ### Get next board list of the board
  def nextBoards(self):
    acts=self.nextActs()
    bs=map(lambda x: self.act(x), acts)
    return bs
