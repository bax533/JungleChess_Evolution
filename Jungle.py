from enum import Enum
import random
import math
import heapq
import time

directions = [(-1,0), (0, 1), (1, 0), (0, -1)]

INF = 1000000.0

P1 = 0
P2 = 1

p2Traps = frozenset([(2, 0), (4, 0), (3, 1)])
p1Traps = frozenset([(2, 8), (4, 8), (3, 7)])

ponds = frozenset([(1, 3), (2, 3), (4, 3), (5, 3),(1, 4), (2, 4), (4, 4), (5, 4),(1, 5), (2, 5), (4, 5), (5, 5)])

p2Cave = (3, 0) #x, y
p1Cave = (3, 8)

RAT = 1
CAT = 2
DOG = 3
WOLF = 4
PANTHER = 5
TIGER = 6
LION = 7
ELEPHANT = 8

class Pawn:
	def __init__(self, typ, pos):
		self.typ = typ
		self.pos = pos

	def prettyType(self, playerNr):
		if(playerNr == P1):
			if(self.typ == RAT):
				return "r"
			elif(self.typ == CAT):
				return "c"
			elif(self.typ == DOG):
				return "d"
			elif(self.typ == WOLF):
				return "w"
			elif(self.typ == PANTHER):
				return "j"
			elif(self.typ == TIGER):
				return "t"
			elif(self.typ == LION):
				return "l"
			elif(self.typ == ELEPHANT):
				return "e"
		else:
			if(self.typ == RAT):
				return "R"
			elif(self.typ == CAT):
				return "C"
			elif(self.typ == DOG):
				return "D"
			elif(self.typ == WOLF):
				return "W"
			elif(self.typ == PANTHER):
				return "J"
			elif(self.typ == TIGER):
				return "T"
			elif(self.typ == LION):
				return "L"
			elif(self.typ == ELEPHANT):
				return "E"

	def __hash__(self):
		return hash(self.typ) ^ hash(self.pos)

def IsOnBoard(pos):
	if(pos[0] >= 0 and pos[0] < 7 and pos[1] >= 0 and pos[1] < 9):
		return True
	return False

def GetStartState():
	p1StartFigures = set([Pawn(RAT, (6, 6)), Pawn(CAT, (1, 7)), Pawn(DOG, (5, 7)), Pawn(WOLF, (2, 6)), Pawn(PANTHER, (4, 6)), Pawn(TIGER, (0,8)), Pawn(LION, (6, 8)), Pawn(ELEPHANT, (0, 6))])
	p2StartFigures = set([Pawn(RAT, (0, 2)), Pawn(CAT, (5, 1)), Pawn(DOG, (1, 1)), Pawn(WOLF, (4, 2)), Pawn(PANTHER, (2, 2)), Pawn(TIGER, (6,0)), Pawn(LION, (0, 0)), Pawn(ELEPHANT, (6, 2))])

	retState = State(p1StartFigures, p2StartFigures, P1, 0)
	return retState

class Game():
	def __init__(self, weightsP1, weightsP2):
		self.weightsP1 = weightsP1
		self.weightsP2 = weightsP2
		self.state = GetStartState()

	def Play(self):
		t = time.time()
		while(not self.state.Terminal()):
			if(self.state.player == P1):
				action = AlphaBeta(self.state, 3, self.weightsP1, -INF, INF)[1]
			else:
				action = AlphaBeta(self.state, 3, self.weightsP2, -INF, INF)[1]
			#print(action)
			self.state = self.state.Succ(action)
		#print(time.time() - t)
		if(self.state.Win(P1)):
			return self.weightsP1
		else:
			return self.weightsP2

class State:
	p1Figures = set()
	p2Figures = set()


	def __init__(self, p1Figures, p2Figures, playerNr, noCapture):
		self.player = playerNr
		self.p1Figures = p1Figures
		self.p2Figures = p2Figures
		self.noCapture = noCapture
		self.wins = 0
		#self.actions = self.GetActions()


	def GetActions(self):
		newActions = set()
		if(self.player == P1):
			for figure in self.p1Figures:
				if(figure.typ == RAT):
					for mov in directions:#wszystkie ruchy
						newPos = (figure.pos[0] + mov[0], figure.pos[1] + mov[1])
						if(not IsOnBoard(newPos)):
							continue

						if(self.IsOnEnemyPawn(newPos)):
							enemyPawn = self.GetEnemyPawnFromPos(newPos)
							if(((enemyPawn.typ == RAT or enemyPawn.typ == ELEPHANT) and not self.IsInWater(figure.pos)) or self.IsInMyTrap(enemyPawn)):
								newActions.add((RAT, newPos))
						elif(not self.IsOnMyPawn(newPos) and not self.IsMyCave(newPos)):
							newActions.add((RAT, newPos))
							
				elif((figure.typ > RAT and figure.typ < TIGER) or figure.typ == ELEPHANT):
					for mov in directions:#wszystkie ruchy
						newPos = (figure.pos[0] + mov[0], figure.pos[1] + mov[1])
						if(not IsOnBoard(newPos)):
							continue

						if(self.IsOnEnemyPawn(newPos)):
							enemyPawn = self.GetEnemyPawnFromPos(newPos)
							if((enemyPawn.typ <= figure.typ and not self.IsInWater(enemyPawn.pos)) or self.IsInMyTrap(enemyPawn)):
								newActions.add((figure.typ, newPos))
						elif(not self.IsOnMyPawn(newPos) and not self.IsMyCave(newPos) and not self.IsInWater(newPos)):
							newActions.add((figure.typ, newPos))
				else:#tiger or lion
					for mov in directions:#wszystkie ruchy
						newPos = (figure.pos[0] + mov[0], figure.pos[1] + mov[1])
						if(not IsOnBoard(newPos)):
							continue

						if(self.IsOnEnemyPawn(newPos)):
							enemyPawn = self.GetEnemyPawnFromPos(newPos)
							if((enemyPawn.typ <= figure.typ and not self.IsInWater(enemyPawn.pos)) or self.IsInMyTrap(enemyPawn)):
								newActions.add((figure.typ, newPos))
						elif((not self.IsOnMyPawn(newPos)) and (not self.IsMyCave(newPos)) and (not self.IsInWater(newPos))):
							newActions.add((figure.typ, newPos))
						elif(self.IsInWater(newPos)):
							wasRat = False
							while(self.IsInWater(newPos)):
								if(self.IsEnemyRat(newPos)):
									wasRat = True
									break
								newPos = (newPos[0] + mov[0], newPos[1] + mov[1])
								
							if(not wasRat):
								newPawn = Pawn(figure.typ, newPos)
								if(self.IsOnEnemyPawn(newPos)):
									enemyPawn = self.GetEnemyPawnFromPos(newPos)
									if(enemyPawn.typ <= figure.typ):
										newActions.add((figure.typ, newPos))
								else:
									newActions.add((figure.typ, newPos))
		else: #P2
			for figure in self.p2Figures:
				if(figure.typ == RAT):
					for mov in directions:#wszystkie ruchy
						newPos = (figure.pos[0] + mov[0], figure.pos[1] + mov[1])
						if(not IsOnBoard(newPos)):
							continue

						if(self.IsOnEnemyPawn(newPos)):
							enemyPawn = self.GetEnemyPawnFromPos(newPos)
							if(((enemyPawn.typ == RAT or enemyPawn.typ == ELEPHANT) and not self.IsInWater(figure.pos)) or self.IsInMyTrap(enemyPawn)):
								newActions.add((figure.typ, newPos))
						elif(not self.IsOnMyPawn(newPos) and not self.IsMyCave(newPos)):
							newActions.add((figure.typ, newPos))
				elif((figure.typ > RAT and figure.typ < TIGER) or figure.typ == ELEPHANT):
					for mov in directions:#wszystkie ruchy
						newPos = (figure.pos[0] + mov[0], figure.pos[1] + mov[1])
						if(not IsOnBoard(newPos)):
							continue

						if(self.IsOnEnemyPawn(newPos)):
							enemyPawn = self.GetEnemyPawnFromPos(newPos)
							if((enemyPawn.typ <= figure.typ and not self.IsInWater(enemyPawn.pos)) or self.IsInMyTrap(enemyPawn)):
								newActions.add((figure.typ, newPos))
						elif(not self.IsOnMyPawn(newPos) and not self.IsMyCave(newPos) and not self.IsInWater(newPos)):
							newActions.add((figure.typ, newPos))
				else:#tiger or lion
					for mov in directions:#wszystkie ruchy
						newPos = (figure.pos[0] + mov[0], figure.pos[1] + mov[1])
						if(not IsOnBoard(newPos)):
							continue

						if(self.IsOnEnemyPawn(newPos)):
							enemyPawn = self.GetEnemyPawnFromPos(newPos)
							if((enemyPawn.typ <= figure.typ and not self.IsInWater(enemyPawn.pos)) or self.IsInMyTrap(enemyPawn)):
								newActions.add((figure.typ, newPos))
						elif(not self.IsOnMyPawn(newPos) and not self.IsMyCave(newPos) and not self.IsInWater(newPos)):
							newActions.add((figure.typ, newPos))
						elif(self.IsInWater(newPos)):
							wasRat = False
							while(self.IsInWater(newPos)):
								if(self.IsEnemyRat(newPos)):
									wasRat = True
									break
								newPos = (newPos[0] + mov[0], newPos[1] + mov[1])
								
							if(not wasRat):
								newPawn = Pawn(figure.typ, newPos)
								if(self.IsOnEnemyPawn(newPos)):
									enemyPawn = self.GetEnemyPawnFromPos(newPos)
									if(enemyPawn.typ <= figure.typ):
										newActions.add((figure.typ, newPos))
								else:
									newActions.add((figure.typ, newPos))
		return newActions		

	def Succ(self, a):
		newP1figures = self.p1Figures.copy()
		newP2figures = self.p2Figures.copy()

		if(self.player == P1):
			for p1figure in newP1figures:
				if(p1figure.typ == a[0]):
					newP1figures.remove(p1figure)
					break

			newP1figures.add(Pawn(a[0], a[1]))

			for figure in self.p2Figures:
				if(figure.pos == a[1]):
					newP2figures.remove(figure)
					return State(newP1figures, newP2figures, 1-self.player, 0)
			
			return State(newP1figures, newP2figures, 1-self.player, self.noCapture + 1)
		else:
			for p2figure in newP2figures:
				if(p2figure.typ == a[0]):
					newP2figures.remove(p2figure)
					break

			newP2figures.add(Pawn(a[0], a[1]))

			for figure in self.p1Figures:
				if(figure.pos == a[1]):
					newP1figures.remove(figure)
					return State(newP1figures, newP2figures, 1-self.player, 0)
			
			return State(newP1figures, newP2figures, 1-self.player, self.noCapture + 1)

	def Heuristic(self, weights):
		retVal = 0.0
		for pawn in self.p1Figures:
			if(pawn.typ == RAT):
				retVal -= weights[0] * dist(pawn.pos, p2Cave)
			elif(pawn.typ == CAT):
				retVal -= weights[1] * dist(pawn.pos, p2Cave)
			elif(pawn.typ == DOG):
				retVal -= weights[2] * dist(pawn.pos, p2Cave)
			elif(pawn.typ == WOLF):
				retVal -= weights[3] * dist(pawn.pos, p2Cave)
			elif(pawn.typ == PANTHER):
				retVal -= weights[4] * dist(pawn.pos, p2Cave)
			elif(pawn.typ == TIGER):
				retVal -= weights[5] * dist(pawn.pos, p2Cave)
			elif(pawn.typ == LION):
				retVal -= weights[6] * dist(pawn.pos, p2Cave)
			elif(pawn.typ == ELEPHANT):
				retVal -= weights[7] * dist(pawn.pos, p2Cave)

		for pawn in self.p2Figures:
			if(pawn.typ == RAT):
				retVal += (1.0 - weights[0]) * dist(pawn.pos, p1Cave)
			elif(pawn.typ == CAT):
				retVal += (1.0 - weights[1]) * dist(pawn.pos, p1Cave)
			elif(pawn.typ == DOG):
				retVal += (1.0 - weights[2]) * dist(pawn.pos, p1Cave)
			elif(pawn.typ == WOLF):
				retVal += (1.0 - weights[3]) * dist(pawn.pos, p1Cave)
			elif(pawn.typ == PANTHER):
				retVal += (1.0 - weights[4]) * dist(pawn.pos, p1Cave)
			elif(pawn.typ == TIGER):
				retVal += (1.0 - weights[5]) * dist(pawn.pos, p1Cave)
			elif(pawn.typ == LION):
				retVal += (1.0 - weights[6]) * dist(pawn.pos, p1Cave)
			elif(pawn.typ == ELEPHANT):
				retVal += (1.0 - weights[7]) * dist(pawn.pos, p1Cave)

		retVal -= weights[8] * self.noCapture

		return retVal

	def IsEnemyRat(self, pos):
		if(self.player == P1):
			for pawn in self.p2Figures:
				if(pawn.pos == pos):
					return True
		else:
			for pawn in self.p1Figures:
				if(pawn.pos == pos):
					return True
		return False

	def GetEnemyPawnFromPos(self, pos):
		if(self.player == P1):
			for enemyPawn in self.p2Figures:
				if(enemyPawn.pos == pos):
					return enemyPawn
		else:
			for enemyPawn in self.p1Figures:
				if(enemyPawn.pos == pos):
					return enemyPawn		

	def IsOnEnemyPawn(self, pos):
		if(self.player == P1):
			for enemyPawn in self.p2Figures:
				if(pos == enemyPawn.pos):
					return True
			return False
		else:
			for enemyPawn in self.p1Figures:
				if(pos == enemyPawn.pos):
					return True
			return False

	def IsOnMyPawn(self, pos):
		if(self.player == P1):
			for pawn in self.p1Figures:
				if(pos == pawn.pos):
					return True
		else:
			for pawn in self.p2Figures:
				if(pos == pawn.pos):
					return True
		return False

	def IsInWater(self, pos):
		if(pos in ponds):
			return True
		else:
			return False

	def IsMyCave(self, pos):
		if(self.player == P1):
			if(pos == p1Cave):
				return True
		else:
			if(pos == p2Cave):
				return True
		return False

	def IsInMyTrap(self, pawn):
		if(self.player == P1):
			if(pawn.pos in p1Traps):
				return True
		else:
			if(pawn.pos in p2Traps):
				return True
		return False

	def Terminal(self):
		for pawn in self.p1Figures:
			if(pawn.pos == p2Cave):
				return True
		for pawn in self.p2Figures:
			if(pawn.pos == p1Cave):
				return True
		if(len(self.p1Figures) == 0 or len(self.p2Figures) == 0):
			return True

		if(self.noCapture >= 30):
			return True
		#if(len(self.GetActions()) == 0):
		#	return True

		return False

	def Utility(self):
		if(self.Win(P1)):
			return 10000
		elif(self.Win(P2)):
			return -10000
		return 0

	def Win(self, playerNr):

		if(self.noCapture >= 30):
			bestP1 = 0
			bestP2 = 0
			for pawn in self.p1Figures:
				if(pawn.typ > bestP1):
					bestP1 = pawn.typ
			for pawn in self.p2Figures:
				if(pawn.typ > bestP2):
					bestP2 = pawn.typ
			#print(bestP2, bestP1)
			if(playerNr == P1):
				if(bestP1 > bestP2):
					return True
				elif(bestP1 == bestP2):
					return False
				else:
					return False
			else:
				if(bestP2 > bestP1):
					return True
				elif(bestP1 == bestP2):
					return True
				else:
					return False

		if(playerNr == P1):
			for pawn in self.p1Figures:
				if(pawn.pos == p2Cave):
					return True
			if(len(self.p2Figures) == 0):
				return True
		else:
			for pawn in self.p2Figures:
				if(pawn.pos == p1Cave):
					return True
			if(len(self.p1Figures) == 0):
				return True
		return False
		

	def SmthInPond(self):
		for pawn in self.p1Figures:
			if(pawn.pos in ponds and pawn.typ != RAT):
				return True
		for pawn in self.p2Figures:
			if(pawn.pos in ponds and pawn.typ != RAT):
				return True

	def PrettyPrint(self):
		for y in range(0, 9):
			printString = ""
			for x in range(0, 7):
				curPos = (x, y)
				added = False
				for pawn in self.p1Figures:
					if(pawn.pos == curPos):
						printString += pawn.prettyType(P1)
						added = True
				for pawn in self.p2Figures:
					if(pawn.pos == curPos):
						printString += pawn.prettyType(P2)
						added = True
				if(added):
					continue
				if(curPos in ponds):
					printString += "~"
				elif(curPos == p1Cave or curPos == p2Cave):
					printString += "*"
				elif(curPos in p1Traps or curPos in p2Traps):
					printString += "#"
				else:
					printString += "."

			print(printString)
		print("-------------")



	def __hash__(self):
		return hash(frozenset(self.p1Figures)) ^ hash(frozenset(self.p2Figures))


def RandomWeights():
	return tuple([random.uniform(0,1) for it in range(9)])

def population(count):
	return [RandomWeights() for x in range(count)]

def breed(mother, father):
	if(mother != father):
		child = []
		for it in range(len(mother)):
			mw = mother[it]
			fw = father[it]

			if(fw > mw):
				childWeight = random.uniform(mw * 0.95, fw * 1.05)
			else:
				childWeight = random.uniform(fw * 0.95, mw * 1.05)

			if(childWeight > 1.0):
				childWeight = 1.0
			if(childWeight < 0.0):
				childWeight = 0.0
			child.append(childWeight)
		return tuple(child)
	else:
		print("mama = tata   ERROR")

def mutate(agent):
	mutatedAgent = []
	for w in agent:
		if(w < 0.5):
			newW = (1 - w) + random.uniform(-0.5, 0.1)
		else:
			newW = (1 - w) + random.uniform(-0.1, 0.5)

		if(newW < 0.0):
			newW = 0.0
		if(newW > 1.0):
			newW = 1.0

		mutatedAgent.append(newW)
	return tuple(mutatedAgent)

def evolve(pop, gamesFactor = 2, retain = 0.2, randomSelect = 0.05, mutateChance = 0.02):
	agentScore = {}
	numGames = len(pop) * gamesFactor

	for it in range(numGames):
		#print((['|' for x in range(it+1)]), end = '')
		#print("")
		competitors = random.sample(pop, 2)
		game = Game(competitors[0], competitors[1])
		winner = game.Play()
		competitors.remove(winner)
		loser = competitors[0]

		if(tuple(winner) not in agentScore):
			agentScore[tuple(winner)] = 1
		else:
			agentScore[tuple(winner)] += 1

		if(tuple(loser) not in agentScore):
			agentScore[tuple(loser)] = -1
		else:
			agentScore[tuple(loser)] -= 1

	topPerformers_size = int(retain * len(pop))
	bottomPerformers_size = len(pop) - topPerformers_size
	randSelect_size = int(len(pop) * randomSelect)
	topPerformers = heapq.nlargest(topPerformers_size, agentScore, key = agentScore.get)
	bottomPerformers = heapq.nsmallest(bottomPerformers_size, agentScore, key = agentScore.get)
	parents = topPerformers + random.sample(bottomPerformers, randSelect_size)
	random.shuffle(parents)

	#Creating children

	numChildren = len(pop) - len(parents)
	children = []
	for it in range(numChildren):
		par = random.sample(parents, 2)
		father = par[0]
		mother = par[1]

		while(father == mother):
			par = random.sample(parents, 2)
			father = par[0]
			mother = par[1]

		child = breed(mother, father)
		children.append(child)

	newPop = parents + children

	mutatedPop = []
	for agent in newPop:
		if(random.uniform(0,1) <= mutateChance):
			mutated = mutate(agent)
			mutatedPop.append(mutated)
		else:
			mutatedPop.append(agent)

	return mutatedPop

#p1 dół planszy - większe numerki
N = 0
def PlayRandomGame(state):
	global N

	N += 1
	while(not state.Terminal()):
		actions = state.GetActions()
		if(len(actions) == 0):
			state = State(state.p1Figures, state.p2Figures, 1-state.player)
		else:
			state = random.choice(tuple(actions))
		N += 1
	if(state.Win(P1)):
		return True
	else:
		return False


def AlphaBeta(state, d, weights, alpha, beta):
	if(state.Terminal()):
		return (state.Utility(), 'none')

	if(d == 0):
		return (state.Heuristic(weights), 'none')

	if(state.player == P1):
		bestVal = -INF
		bestAct = (0, (0,0))
		for s in [(state.Succ(action), action) for action in state.GetActions()]:
			result = AlphaBeta(s[0], d-1, weights, alpha, beta)
			value = result[0]

			if(value > bestVal):
				bestVal = value
				bestAct = s[1]

			alpha = max(alpha, bestVal)
			if(beta <= alpha):
				break
		#print(bestAct)
		return (bestVal, bestAct)
	else:
		bestVal = INF
		bestAct = (0, (0,0))
		for s in [(state.Succ(action), action) for action in state.GetActions()]:
			#print("dupa")
			result = AlphaBeta(s[0], d-1, weights, alpha, beta)
			value = result[0]
			if(value < bestVal):
				bestVal = value
				bestAct = s[1]

			beta = min(beta, bestVal)
			if(beta <= alpha):
				break
		#print(bestAct)
		return (bestVal, bestAct)


def MinMax(state, d, weights):

	def recurse(state, d):
		if(state.Terminal()):
			return (state.Utility(), 'none')

		if(d == 0):
			return (state.Heuristic(weights), 'none')

		choices = [(recurse(state.Succ(action), d-1)[0], action) for action in state.GetActions()]
		if(state.player == P1):
			return max(choices)
		else:
			return min(choices)
	value, action = recurse(state, d)
	#print(value)
	return action

def dist(p1, p2):
	return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def f(state):
	retVal = 0
	bestDist = 1000
	if(state.player == P1):
		for pawn in state.p1Figures:
			if(dist(pawn.pos, p2Cave) < bestDist):
				bestDist = dist(pawn.pos, p2Cave)

			#bestDist += dist(pawn.pos, p2Cave)

			retVal += pawn.typ/2

			if(pawn.typ == RAT or pawn.typ == TIGER or pawn.typ == LION):
				retVal += 2
	else:
		for pawn in state.p2Figures:
			if(dist(pawn.pos, p1Cave) < bestDist):
				bestDist = dist(pawn.pos, p1Cave)

			#bestDist += dist(pawn.pos, p1Cave)

			retVal += pawn.typ/2

			if(pawn.typ == RAT or pawn.typ == TIGER or pawn.typ == LION):
				retVal += 2

	return bestDist# + retVal


def PickBest(pop):
	agentScore = {}
	numGames = len(pop) * 5

	for it in range(numGames):
		#print((['|' for x in range(it+1)]), end = '')
		#print("")
		competitors = random.sample(pop, 2)
		game = Game(competitors[0], competitors[1])
		winner = game.Play()
		competitors.remove(winner)
		loser = competitors[0]

		if(tuple(winner) not in agentScore):
			agentScore[tuple(winner)] = 1
		else:
			agentScore[tuple(winner)] += 1

		if(tuple(loser) not in agentScore):
			agentScore[tuple(loser)] = -1
		else:
			agentScore[tuple(loser)] -= 1

	topPerformers = heapq.nlargest(5, agentScore, key = agentScore.get)
	retAgents = []
	for agent in topPerformers:
		retAgents.append(agent)

	return retAgents


def PerformEvolution(populationCount = 100, evolutionCycles = 15):
	#populationCount = 100
	#evolutionCycles = 25
	pop = population(populationCount)
	for i in range(evolutionCycles):
		print(i)
		pop = evolve(pop, gamesFactor = 5, retain = 0.2, randomSelect = 0.05, mutateChance = 0.05)

	topAgents = PickBest(pop)

	f = open("latestPop.txt", "w")
	for agent in pop:
		wStr = ""
		for w in agent:
			wStr += str(w)
		wStr+="\n"
		f.write(wStr)	
	f.close()

	f = open("bestChildren.txt", "w")
	for agent in topAgents:
		f.write(str(agent) + "\n")
	f.close()

	return topAgents[0]

num_games = 100
wonGames = 0

#loadedWeights = (0.38419572203145924, 0.4485672936569485, 0.41694046780787186, 0.5656723820812245, 0.5179563685436197, 0.3934933597945433, 0.5937341570059343, 0.5248729544087116)
#loadedWeights = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

loadedWeights = PerformEvolution()
#loadedWeights = (0.29390961424604156, 0.5631026472569237, 0.7481031791056122, 0.5025964205614991, 0.5224287922444969, 0.4119763474690025, 0.6164558898462176, 0.765813196725114, 0.7650963552589066)

for i in range(num_games):
	thisState = GetStartState()

	while(not thisState.Terminal()):
		if(thisState.player == P1):
			action = AlphaBeta(thisState, 4, loadedWeights, -INF, INF)[1]
		else:
			'''
			action = random.choice(tuple(thisState.GetActions()))
			'''
			bestVal = INF
			for a in thisState.GetActions():
				if(f(thisState.Succ(a)) < bestVal):
					bestVal = f(thisState.Succ(a))
					action = a

		thisState = thisState.Succ(action)
		#thisState.PrettyPrint()
		#print(action)
	#thisState.PrettyPrint()
	if(thisState.Win(P1)):
		wonGames += 1
	print(wonGames, "/", i+1)

#print(wonGames)
#node.parent.thisState.PrettyPrint()
#node.thisState.PrettyPrint()
