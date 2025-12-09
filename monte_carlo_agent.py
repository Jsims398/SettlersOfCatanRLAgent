import random
import numpy as np
from copy import deepcopy


DICE_PROB = {
    2: 1/36,
    3: 2/36,
    4: 3/36,
    5: 4/36,
    6: 5/36,
    8: 5/36,
    9: 4/36,
    10: 3/36,
    11: 2/36,
    12: 1/36
}


class MonteCarloSettler:
    def __init__(
        self,
        board,
        players,
        num_simulations=500,
        weight_resource=1.0,
        weight_prob=1.0,
        weight_diversity=1.0,
        weight_blocking=0.6
    ):
        self.base_board = board
        self.players = players
        self.num_simulations = num_simulations

        # heuristic weights
        self.W_RES = weight_resource
        self.W_PROB = weight_prob
        self.W_DIV = weight_diversity
        self.W_BLOCK = weight_blocking

    # ------------------------------------------------------------
    # PUBLIC Calls
    # ------------------------------------------------------------
    def choose_best_settlement_setup(self, player):
        valid_spots = self.base_board.get_setup_settlements(player)

        if not valid_spots:
            return None

        scores = {}
        for v in valid_spots:
            scores[v] = self.run_rollouts_for_settlement(player, v)

        return max(scores, key=scores.get)
    
    def choose_best_settlement_potential(self, player):
        valid_spots = self.base_board.get_potential_settlements(player)

        if not valid_spots:
            return None

        scores = {}
        for v in valid_spots:
            scores[v] = self.run_rollouts_for_settlement(player, v)

        return max(scores, key=scores.get)

    def choose_best_road(self, player):
        valid_edges = self.base_board.get_legal_road_edges(player)

        if not valid_edges:
            return None

        scores = {}
        for e in valid_edges:
            scores[e] = self.evaluate_road(player, e)

        return max(scores, key=scores.get)

    # ------------------------------------------------------------
    # MONTE CARLO ROLLOUT FOR SETTLEMENT ONLY
    # ------------------------------------------------------------
    def run_rollouts_for_settlement(self, player, vertex):
        scores = []
        for _ in range(self.num_simulations):
            # deep copy board and player for isolated simulation
            sim_board = deepcopy(self.base_board)
            sim_player = deepcopy(player)

            # apply candidate settlement via player method
            sim_player.build_settlement(vertex, sim_board)

            # score using heuristic + sampling variations
            score = self.evaluate_settlement(sim_board, vertex, sim_player)

            scores.append(score)

        return np.mean(scores)

    # ------------------------------------------------------------
    # HEURISTIC EVALUATION
    # ------------------------------------------------------------

    
    def evaluate_settlement(self, board, vertex, player):
        adj_hexes = board.boardGraph[vertex].adjacentHexList

        resource_counts = {}
        production_score = 0.0

        for h in adj_hexes:
            tile = board.hexTileDict[h]

            if tile.resource.type == "DESERT":
                continue

            res = tile.resource.type
            resource_counts[res] = resource_counts.get(res, 0) + 1

            if tile.resource.num in DICE_PROB:
                production_score += DICE_PROB[tile.resource.num]

        abundance_score = sum(resource_counts.values())
        diversity_score = len(resource_counts)

        # heuristic combination
        return (
            self.W_RES * abundance_score +
            self.W_PROB * production_score +
            self.W_DIV * diversity_score
        )


    # ------------------------------------------------------------
    # ROAD EVALUATION (FUTURE POTENTIAL + BLOCKING)
    # ------------------------------------------------------------
    def evaluate_road(self, player, edge):
        endpoints = self.base_board.get_vertices_adjacent_to_edge(edge)
        best_future_spot = 0

        # value = best reachable settlement from edge
        for v in endpoints:
            if v in self.base_board.get_potential_settlements(player):
                val = self.evaluate_settlement(self.base_board, v, player)
                best_future_spot = max(best_future_spot, val)

        blocking_score = self.estimate_blocking_value(player, edge)

        return best_future_spot + self.W_BLOCK * blocking_score

    def estimate_blocking_value(self, player, edge):
        endpoints = self.base_board.get_vertices_adjacent_to_edge(edge)
        worst_loss = 0

        for other in self.players:
            if other == player:
                continue
            for v in endpoints:
                if v in self.get_valid_settlements(self.base_board, other):
                    val = self.evaluate_settlement(self.base_board, v, other)
                    worst_loss = max(worst_loss, val)

        return worst_loss