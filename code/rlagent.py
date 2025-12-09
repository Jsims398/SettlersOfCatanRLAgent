from board import *
from player import *

class RLAgent:
    def __init__(self, board):
        self.board = board

    def choose_best_settlement_setup(self, player):
        options = list(self.board.get_setup_settlements(player))
        best_v = None
        best_score = -999

        for v in options:
            # pass player so proximity bonuses (relative to player's last settlement)
            # are considered by the board heuristic
            score = self.board.evaluate_settlement_reward(v, player)
            if score > best_score:
                best_score = score
                best_v = v
        self.best_v = best_v
        return best_v
    
    def choose_best_road(self, player):
        """
        Chooses the best road placement for the given player, considering the player’s next settlement move 
        and blocking other players from potential expansions.
        """
        possible_roads = self.board.get_setup_roads(player)
        best_road = None
        best_value = -float('inf')

        for road in possible_roads:
            value = self.get_closest_best_settlement(player, road)

            if value > best_value:
                best_value = value
                best_road = road

        return best_road


    def get_closest_best_settlement(self, player, possible_spots):
        score = 0

        adjacent_settlements = self.board.get_adjacent_settlement_spots(self.best_v)
        settlementScoreList = []
        for settlement in adjacent_settlements:
            # include player so proximity bonus (to player's last settlement) is included
            settlementScore = self.board.evaluate_settlement_reward(settlement, player)
            # additionally include any road-based proximity heuristics
            proximityBonus = self.board.evaluate_proximity_to_settlements(player, settlement)
            settlementScoreList.append(settlementScore + proximityBonus)

        return max(settlementScoreList) if settlementScoreList else 0

        