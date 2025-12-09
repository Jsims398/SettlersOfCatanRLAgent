from board import catanBoard
from player import player
import random
from collections import deque
import copy


class CatanSetupEnv:
    def __init__(self, num_players: int = 4, agent_index: int = 0, board_template=None):
        self.num_players = num_players
        self.agent_index = agent_index
        self.board_template = board_template
        if self.board_template is not None:
            try:
                self.n_actions = len(self.board_template.vertex_index_to_pixel_dict)
            except Exception:
                proto = catanBoard()
                self.n_actions = len(proto.vertex_index_to_pixel_dict)
        else:
            proto = catanBoard()
            self.n_actions = len(proto.vertex_index_to_pixel_dict)

        self.board = None
        self.players = None
        self.turn_sequence = None
        self.turn_ptr = 0
        self.agent_action_count = 0

    def get_num_actions(self):
        return self.n_actions

    def vertex_to_index(self, v_pixel):
        for idx, pix in self.board.vertex_index_to_pixel_dict.items():
            if pix == v_pixel:
                return idx
        raise KeyError('vertex pixel not found')

    def index_to_vertex(self, idx):
        return self.board.vertex_index_to_pixel_dict[idx]

    def legal_settlements(self, p):
        sdict = self.board.get_setup_settlements(p)
        spots = []
        for v_pixel in sdict.keys():
            try:
                idx = self.vertex_to_index(v_pixel)
                spots.append(idx)
            except KeyError:
                continue
        return spots

    def reset(self):
        if self.board_template is not None:
            try:
                self.board = copy.deepcopy(self.board_template)
            except Exception:
                self.board = catanBoard()
        else:
            self.board = catanBoard()
        colors = ['black', 'darkslateblue', 'magenta4', 'orange1']
        self.players = [player(chr(65 + i), colors[i % len(colors)]) for i in range(self.num_players)]

        forward = list(range(self.num_players))
        reverse = list(reversed(forward))
        self.turn_sequence = forward + reverse
        self.turn_ptr = 0
        self.agent_action_count = 0

        return None

    def get_legal_actions(self):
        cur_player_idx = self.turn_sequence[self.turn_ptr]
        cur_player = self.players[cur_player_idx]
        return self.legal_settlements(cur_player)

    def step(self, action: int):
        cur_idx = self.turn_sequence[self.turn_ptr]
        cur_player = self.players[cur_idx]
        info = {}
        reward = 0.0

        if cur_idx == self.agent_index:
            chosen_idx = action
        else:
            legal = self.legal_settlements(cur_player)
            if not legal:
                chosen_idx = None
            else:
                chosen_idx = random.choice(legal)

        if chosen_idx is not None:
            settlement_spot = self.index_to_vertex(chosen_idx)
            cur_player.build_settlement(settlement_spot, self.board)
            roads = self.board.get_setup_roads(cur_player)
            if roads:
                best = None
                best_score = -1e9
                for (v1, v2) in roads.keys():
                    score = self.board.evaluate_road_reward((v1, v2))
                    if score > best_score:
                        best_score = score
                        best = (v1, v2)
                if best is not None:
                    cur_player.build_road(best[0], best[1], self.board)

        done = False
        if cur_idx == self.agent_index:
            try:
                reward = self.board.evaluate_settlement_reward(settlement_spot, cur_player)
            except Exception:
                reward = 0.0

            info['phase'] = self.agent_action_count
            self.agent_action_count += 1

        self.turn_ptr += 1
        if self.turn_ptr >= len(self.turn_sequence):
            done = True

        return None, reward, done, info
