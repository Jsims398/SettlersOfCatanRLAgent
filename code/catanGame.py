#Settlers of Catan
#Gameplay class with pygame

from board import *
from gameView import *
from player import *
import queue
import random
import numpy as np
import sys, pygame
import os
import time
import copy

from rlagent import *
from sarsa_catan import SARSASetupAgent
from catan_env import CatanSetupEnv

#Catan gameplay class definition
class catanGame():
    #Create new gameboard
    def __init__(self):
        print("Initializing Settlers of Catan Board...")
        self.board = catanBoard()
        # keep a template copy of the initial board layout so other agents
        # (e.g., SARSA) can use the exact same board configuration
        try:
            self.initial_board_template = copy.deepcopy(self.board)
        except Exception:
            # fallback: store the board itself (best-effort)
            self.initial_board_template = self.board

        #Game State variables
        self.gameOver = False
        self.maxPoints = 8
        self.numPlayers = 0

        while(self.numPlayers not in [3,4]): #Only accept 3 and 4 player games
            try:
                self.numPlayers = int(input("Enter Number of Players (3 or 4):"))
            except:
                print("Please input a valid number")

        print("Initializing game with {} players...".format(self.numPlayers))
        print("Note that Player 1 goes first, Player 2 second and so forth.")
        
        #Initialize blank player queue and initial set up of roads + settlements
        self.playerQueue = queue.Queue(self.numPlayers)
        self.gameSetup = True #Boolean to take care of setup phase

        #Initialize boardview object
        self.boardView = catanGameView(self.board, self)

        #Run functions to view board and vertex graph
        # self.board.printGraph()
        self.boardView.displayGameScreen()
        input("Press Enter to continue...")

        #Functiont to go through initial set up
        self.build_initial_settlements()

        #Display initial board
        self.boardView.displayGameScreen()
        
    def build_initial_settlements(self):
        # Initialize new players with names and colors
        playerColors = ['black', 'darkslateblue', 'magenta4', 'orange1']
        for i in range(self.numPlayers):
            playerNameInput = ['a', 'b', 'c', 'd'].pop() # input("Enter Player {} name: ".format(i+1))
            newPlayer = player(playerNameInput, playerColors[i])
            self.playerQueue.put(newPlayer)

        playerList = list(self.playerQueue.queue)

        # self.boardView.displayGameScreen()  # Display the initial gameScreen
        # print("Displaying Initial GAMESCREEN!")

        for player_i in playerList:
            rl_agent = RLAgent(self.board) 

            best_settle = rl_agent.choose_best_settlement_setup(player_i)
            # settlement = self.board.get_setup_settlements(player_i).get(best_settle, None)
            print(best_settle)
            self.build(player_i, 'SETTLE', best_settle)

            best_road = rl_agent.choose_best_road(player_i)
            self.build(player_i, 'ROAD', best_road)

            # self.boardView.displayGameScreen()  # Update the screen after each action

        playerList.reverse()  # Reverse the order for the second phase of placement
        for player_i in playerList:
            rl_agent = RLAgent(self.board)

            best_settle = rl_agent.choose_best_settlement_setup(player_i)
            self.build(player_i, 'SETTLE', best_settle)

            best_road = rl_agent.choose_best_road(player_i)
            self.build(player_i, 'ROAD', best_road)

            # self.boardView.displayGameScreen()  # Update the screen after each action
        
        '''
        # Initial resource generation (players collect resources from adjacent hexes)
        # for player_i in playerList:
        #     # Check each adjacent hex to the latest settlement built by the player
        #     for adjacentHex in self.board.boardGraph[player_i.buildGraph['SETTLEMENTS'][-1]].adjacentHexList:
        #         resourceGenerated = self.board.hexTileDict[adjacentHex].resource.type
        #         if resourceGenerated != 'DESERT':
        #             player_i.resources[resourceGenerated] += 1
        #             print(f"{player_i.name} collects 1 {resourceGenerated} from Settlement")
        # self.boardView.displayGameScreen()
        '''
        try:
            fname = os.path.join(os.getcwd(), f"setup.png")
            surf = pygame.display.get_surface()
            if surf is not None:
                pygame.image.save(surf, fname)
                print(f"Saved setup screenshot: {fname}")
        except Exception as _e:
            print('Failed to save setup screenshot:', _e)

        try:
            env = CatanSetupEnv(num_players=self.numPlayers, agent_index=0, board_template=self.initial_board_template)
            sarsa_agent = SARSASetupAgent(env, alpha=0.1, gamma=0.95, epsilon=0.2)
            q = sarsa_agent.train(num_episodes=2000, max_steps_per_episode=20)
            
            try:
                sarsa_board = copy.deepcopy(self.initial_board_template)
            except Exception:
                sarsa_board = catanBoard()
            sarsa_players = []
            colors = ['black', 'darkslateblue', 'magenta4', 'orange1']
            for i in range(self.numPlayers):
                p = player(['a','b','c','d'][i], colors[i])
                sarsa_players.append(p)

            def best_legal_action_for_phase(board_obj, q_table, phase, legal_indices):
                if not legal_indices:
                    return None
                order = list(reversed(sorted(range(len(q_table[phase])), key=lambda a: q_table[phase][a])))
                for a in order:
                    if a in legal_indices:
                        return a
                return random.choice(legal_indices)

            for idx in range(self.numPlayers):
                cur_p = sarsa_players[idx]
                sdict = sarsa_board.get_setup_settlements(cur_p)
                legal = []
                for v_pixel in sdict.keys():
                    for ii, pix in sarsa_board.vertex_index_to_pixel_dict.items():
                        if pix == v_pixel:
                            legal.append(ii)
                            break
                a = best_legal_action_for_phase(sarsa_board, q, 0, legal)
                if a is not None:
                    v_pix = sarsa_board.vertex_index_to_pixel_dict[a]
                    cur_p.build_settlement(v_pix, sarsa_board)
                    roads = sarsa_board.get_setup_roads(cur_p)
                    if roads:
                        best_road = None
                        best_score = -1e9
                        for (v1, v2) in roads.keys():
                            score = sarsa_board.evaluate_road_reward((v1, v2))
                            if score > best_score:
                                best_score = score
                                best_road = (v1, v2)
                        if best_road:
                            cur_p.build_road(best_road[0], best_road[1], sarsa_board)

            for idx in reversed(range(self.numPlayers)):
                cur_p = sarsa_players[idx]
                sdict = sarsa_board.get_setup_settlements(cur_p)
                legal = []
                for v_pixel in sdict.keys():
                    for ii, pix in sarsa_board.vertex_index_to_pixel_dict.items():
                        if pix == v_pixel:
                            legal.append(ii)
                            break
                a = best_legal_action_for_phase(sarsa_board, q, 1, legal)
                if a is not None:
                    v_pix = sarsa_board.vertex_index_to_pixel_dict[a]
                    cur_p.build_settlement(v_pix, sarsa_board)
                    roads = sarsa_board.get_setup_roads(cur_p)
                    if roads:
                        best_road = None
                        best_score = -1e9
                        for (v1, v2) in roads.keys():
                            score = sarsa_board.evaluate_road_reward((v1, v2))
                            if score > best_score:
                                best_score = score
                                best_road = (v1, v2)
                        if best_road:
                            cur_p.build_road(best_road[0], best_road[1], sarsa_board)

            self.sarsaBoardView = catanGameView(sarsa_board, self)
            self.sarsaBoardView.displayGameScreen()


            try:
                sarsa_fname = os.path.join(os.getcwd(), f"sarsa_setup.png")
                print("saved sarsa screenshot")
                surf2 = pygame.display.get_surface()
                if surf2 is not None:
                    pygame.image.save(surf2, sarsa_fname)
            except Exception as _e:
                print('Failed to save SARSA screenshot:', _e)
        except Exception as e:
            print('SARSA training/display failed:', e)

        self.gameSetup = False



    def build(self, player, build_flag, location):
        """
        Executes an AI-chosen build without UI interaction.
        """
        # if best_vertex is None:
        #     legal_vertices = list(self.board.get_setup_settlements(player).keys())
        #     if not legal_vertices:
        #         return None
        #     best_vertex = random.choice(legal_vertices)
        # return best_vertex


        if build_flag == 'SETTLE':
            player.build_settlement(location, self.board, )

        elif build_flag == 'ROAD':
            v1, v2 = location
            player.build_road(v1, v2, self.board)

        elif build_flag == 'CITY':
            player.build_city(location, self.board)

        # After any build, refresh UI
        # self.boardView.displayGameScreen()
'''

    # def build(self, player, build_flag):
    #     if(build_flag == 'ROAD'): #Show screen with potential roads
    #         if(self.gameSetup):
    #             potentialRoadDict = self.board.get_setup_roads(player)
    #         else:
    #             potentialRoadDict = self.board.get_potential_roads(player)

    #         roadToBuild = self.boardView.buildRoad_display(player, potentialRoadDict)
    #         if(roadToBuild != None):
    #             player.build_road(roadToBuild[0], roadToBuild[1], self.board)

            
    #     if(build_flag == 'SETTLE'): #Show screen with potential settlements
    #         if(self.gameSetup):
    #             potentialVertexDict = self.board.get_setup_settlements(player)
    #         else:
    #             potentialVertexDict = self.board.get_potential_settlements(player)
            
    #         vertexSettlement = self.boardView.buildSettlement_display(player, potentialVertexDict)
    #         if(vertexSettlement != None):
    #             player.build_settlement(vertexSettlement, self.board) 

    #     if(build_flag == 'CITY'): 
    #         potentialCityVertexDict = self.board.get_potential_cities(player)
    #         vertexCity = self.boardView.buildSettlement_display(player, potentialCityVertexDict)
    #         if(vertexCity != None):
    #             player.build_city(vertexCity, self.board) 


    #Wrapper Function to handle robber functionality
    def robber(self, player):
        potentialRobberDict = self.board.get_robber_spots()
        print("Move Robber!")

        hex_i, playerRobbed = self.boardView.moveRobber_display(player, potentialRobberDict)
        player.move_robber(hex_i, self.board, playerRobbed)


    #Function to roll dice 
    def rollDice(self):
        dice_1 = np.random.randint(1,7)
        dice_2 = np.random.randint(1,7)
        diceRoll = dice_1 + dice_2
        print("Dice Roll = ", diceRoll, "{", dice_1, dice_2, "}")

        self.boardView.displayDiceRoll(diceRoll)

        return diceRoll

    #Function to update resources for all players
    def update_playerResources(self, diceRoll, currentPlayer):
        if(diceRoll != 7): #Collect resources if not a 7
            #First get the hex or hexes corresponding to diceRoll
            hexResourcesRolled = self.board.getHexResourceRolled(diceRoll)
            #print('Resources rolled this turn:', hexResourcesRolled)

            #Check for each player
            for player_i in list(self.playerQueue.queue):
                #Check each settlement the player has
                for settlementCoord in player_i.buildGraph['SETTLEMENTS']:
                    for adjacentHex in self.board.boardGraph[settlementCoord].adjacentHexList: #check each adjacent hex to a settlement
                        if(adjacentHex in hexResourcesRolled and self.board.hexTileDict[adjacentHex].robber == False): #This player gets a resource if hex is adjacent and no robber
                            resourceGenerated = self.board.hexTileDict[adjacentHex].resource.type
                            player_i.resources[resourceGenerated] += 1
                            print("{} collects 1 {} from Settlement".format(player_i.name, resourceGenerated))
                
                #Check each City the player has
                for cityCoord in player_i.buildGraph['CITIES']:
                    for adjacentHex in self.board.boardGraph[cityCoord].adjacentHexList: #check each adjacent hex to a settlement
                        if(adjacentHex in hexResourcesRolled and self.board.hexTileDict[adjacentHex].robber == False): #This player gets a resource if hex is adjacent and no robber
                            resourceGenerated = self.board.hexTileDict[adjacentHex].resource.type
                            player_i.resources[resourceGenerated] += 2
                            print("{} collects 2 {} from City".format(player_i.name, resourceGenerated))

                print("Player:{}, Resources:{}, Points: {}".format(player_i.name, player_i.resources, player_i.victoryPoints))
                #print('Dev Cards:{}'.format(player_i.devCards))
                #print("RoadsLeft:{}, SettlementsLeft:{}, CitiesLeft:{}".format(player_i.roadsLeft, player_i.settlementsLeft, player_i.citiesLeft))
                print('MaxRoadLength:{}, LongestRoad:{}\n'.format(player_i.maxRoadLength, player_i.longestRoadFlag))
        
        #Logic for a 7 roll
        else:
            #Implement discarding cards
            #Check for each player
            for player_i in list(self.playerQueue.queue):
                #Player must discard resources
                player_i.discardResources()

            self.robber(currentPlayer)
            self.boardView.displayGameScreen()#Update back to original gamescreen


    #function to check if a player has the longest road - after building latest road
    def check_longest_road(self, player_i):
        if(player_i.maxRoadLength >= 5): #Only eligible if road length is at least 5
            longestRoad = True
            for p in list(self.playerQueue.queue):
                if(p.maxRoadLength >= player_i.maxRoadLength and p != player_i): #Check if any other players have a longer road
                    longestRoad = False
            
            if(longestRoad and player_i.longestRoadFlag == False): #if player_i takes longest road and didn't already have longest road
                #Set previous players flag to false and give player_i the longest road points
                prevPlayer = ''
                for p in list(self.playerQueue.queue):
                    if(p.longestRoadFlag):
                        p.longestRoadFlag = False
                        p.victoryPoints -= 2
                        prevPlayer = 'from Player ' + p.name
    
                player_i.longestRoadFlag = True
                player_i.victoryPoints += 2

                print("Player {} takes Longest Road {}".format(player_i.name, prevPlayer))

    #function to check if a player has the largest army - after playing latest knight
    def check_largest_army(self, player_i):
        if(player_i.knightsPlayed >= 3): #Only eligible if at least 3 knights are player
            largestArmy = True
            for p in list(self.playerQueue.queue):
                if(p.knightsPlayed >= player_i.knightsPlayed and p != player_i): #Check if any other players have more knights played
                    largestArmy = False
            
            if(largestArmy and player_i.largestArmyFlag == False): #if player_i takes largest army and didn't already have it
                #Set previous players flag to false and give player_i the largest points
                prevPlayer = ''
                for p in list(self.playerQueue.queue):
                    if(p.largestArmyFlag):
                        p.largestArmyFlag = False
                        p.victoryPoints -= 2
                        prevPlayer = 'from Player ' + p.name
    
                player_i.largestArmyFlag = True
                player_i.victoryPoints += 2

                print("Player {} takes Largest Army {}".format(player_i.name, prevPlayer))


    #Function that runs the main game loop with all players and pieces
    def playCatan(self):
        #self.board.displayBoard() #Display updated board

        while (self.gameOver == False):

            #Loop for each player's turn -> iterate through the player queue
            for currPlayer in self.playerQueue.queue:

                print("---------------------------------------------------------------------------")
                print("Current Player:", currPlayer.name)

                turnOver = False #boolean to keep track of turn
                diceRolled = False  #Boolean for dice roll status
                
                #Update Player's dev card stack with dev cards drawn in previous turn and reset devCardPlayedThisTurn
                currPlayer.updateDevCards()
                currPlayer.devCardPlayedThisTurn = False

                while(turnOver == False):

                    
                    for e in pygame.event.get(): #Get player actions/in-game events
                        #print(e)
                        if e.type == pygame.QUIT:
                            sys.exit(0)

                        #Check mouse click in rollDice
                        if(e.type == pygame.MOUSEBUTTONDOWN):
                            #Check if player rolled the dice
                            if(self.boardView.rollDice_button.collidepoint(e.pos)):
                                if(diceRolled == False): #Only roll dice once
                                    diceNum = self.rollDice()
                                    diceRolled = True
                                    
                                    self.boardView.displayDiceRoll(diceNum)
                                    #Code to update player resources with diceNum
                                    self.update_playerResources(diceNum, currPlayer)

                            #Check if player wants to build road
                            if(self.boardView.buildRoad_button.collidepoint(e.pos)):
                                #Code to check if road is legal and build
                                if(diceRolled == True): #Can only build after rolling dice
                                    self.build(currPlayer, 'ROAD')
                                    self.boardView.displayGameScreen()#Update back to original gamescreen

                                    #Check if player gets longest road and update Victory points
                                    self.check_longest_road(currPlayer)
                                    #Show updated points and resources  
                                    print("Player:{}, Resources:{}, Points: {}".format(currPlayer.name, currPlayer.resources, currPlayer.victoryPoints))

                            #Check if player wants to build settlement
                            if(self.boardView.buildSettlement_button.collidepoint(e.pos)):
                                if(diceRolled == True): #Can only build settlement after rolling dice
                                    self.build(currPlayer, 'SETTLE')
                                    self.boardView.displayGameScreen()#Update back to original gamescreen
                                    #Show updated points and resources  
                                    print("Player:{}, Resources:{}, Points: {}".format(currPlayer.name, currPlayer.resources, currPlayer.victoryPoints))

                            #Check if player wants to build city
                            if(self.boardView.buildCity_button.collidepoint(e.pos)):
                                if(diceRolled == True): #Can only build city after rolling dice
                                    self.build(currPlayer, 'CITY')
                                    self.boardView.displayGameScreen()#Update back to original gamescreen
                                    #Show updated points and resources  
                                    print("Player:{}, Resources:{}, Points: {}".format(currPlayer.name, currPlayer.resources, currPlayer.victoryPoints))

                            #Check if player wants to draw a development card
                            if(self.boardView.devCard_button.collidepoint(e.pos)):
                                if(diceRolled == True): #Can only draw devCard after rolling dice
                                    currPlayer.draw_devCard(self.board)
                                    #Show updated points and resources  
                                    print("Player:{}, Resources:{}, Points: {}".format(currPlayer.name, currPlayer.resources, currPlayer.victoryPoints))
                                    print('Available Dev Cards:', currPlayer.devCards)

                            #Check if player wants to play a development card - can play devCard whenever after rolling dice
                            if(self.boardView.playDevCard_button.collidepoint(e.pos)):
                                    currPlayer.play_devCard(self)
                                    self.boardView.displayGameScreen()#Update back to original gamescreen
                                    
                                    #Check for Largest Army and longest road
                                    self.check_largest_army(currPlayer)
                                    self.check_longest_road(currPlayer)
                                    #Show updated points and resources  
                                    print("Player:{}, Resources:{}, Points: {}".format(currPlayer.name, currPlayer.resources, currPlayer.victoryPoints))
                                    print('Available Dev Cards:', currPlayer.devCards)

                            #Check if player wants to trade with the bank
                            if(self.boardView.tradeBank_button.collidepoint(e.pos)):
                                    currPlayer.initiate_trade(self, 'BANK')
                                    #Show updated points and resources  
                                    print("Player:{}, Resources:{}, Points: {}".format(currPlayer.name, currPlayer.resources, currPlayer.victoryPoints))
                            
                            #Check if player wants to trade with another player
                            if(self.boardView.tradePlayers_button.collidepoint(e.pos)):
                                    currPlayer.initiate_trade(self, 'PLAYER')
                                    #Show updated points and resources  
                                    print("Player:{}, Resources:{}, Points: {}".format(currPlayer.name, currPlayer.resources, currPlayer.victoryPoints))

                            #Check if player wants to end turn
                            if(self.boardView.endTurn_button.collidepoint(e.pos)):
                                if(diceRolled == True): #Can only end turn after rolling dice
                                    print("Ending Turn!")
                                    turnOver = True  #Update flag to nextplayer turn

                    #Update the display
                    #self.displayGameScreen(None, None)
                    pygame.display.update()
                    
                    #Check if game is over
                    if currPlayer.victoryPoints >= self.maxPoints:
                        self.gameOver = True
                        self.turnOver = True
                        print("====================================================")
                        print("PLAYER {} WINS!".format(currPlayer.name))
                        print("Exiting game in 10 seconds...")
                        break

                if(self.gameOver):
                    startTime = pygame.time.get_ticks()
                    runTime = 0
                    while(runTime < 10000): #10 second delay prior to quitting
                        runTime = pygame.time.get_ticks() - startTime

                    break
                              
'''
#Initialize new game and run
newGame = catanGame()
# newGame.playCatan()
        