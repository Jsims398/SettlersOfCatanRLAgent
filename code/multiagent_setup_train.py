import copy
import random
import numpy as np
from collections import deque
from board import catanBoard
from player import player

try:
    from gameView import catanGameView
    PYGAME_AVAILABLE = True
except Exception:
    PYGAME_AVAILABLE = False

class SARSAAgentPlayer:
    def __init__(self, n_actions, n_states=18, alpha=0.1, gamma=0.95, eps=0.2):
        """
        n_actions: number of possible vertex indices on the board
        n_states: small discrete state space size (we use 18 by default)
        """
        self.n_actions = int(n_actions)
        self.n_states = int(n_states)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = eps

        # Q: states x actions
        self.Q = np.zeros((self.n_states, self.n_actions), dtype=float)
        self.visit_counts = np.zeros((self.n_states, self.n_actions), dtype=int)

        # For SARSA-on-policy updates we store the last (s, a, r) for this agent
        self.last_state = None
        self.last_action = None
        self.last_reward = None

    def choose_action(self, state_id, legal_actions):
        """
        Epsilon-greedy choose action among legal action indices (list of ints).
        """
        legal = list(legal_actions)
        if not legal:
            return None
        if random.random() < self.epsilon:
            return int(random.choice(legal))

        qvals = self.Q[state_id]
        # choose best among legal
        best_val = max(qvals[a] for a in legal)
        bests = [a for a in legal if np.isclose(qvals[a], best_val)]
        return int(random.choice(bests))

    def update_sarsa(self, prev_state, prev_action, prev_reward, next_state, next_action, terminal=False):
        """
        Perform SARSA update for (prev_state, prev_action) using reward prev_reward and
        next_state/next_action (a' may be None if terminal).
        """
        if prev_state is None or prev_action is None:
            return

        self.visit_counts[prev_state, prev_action] += 1
        visits = self.visit_counts[prev_state, prev_action]
        alpha_t = self.alpha / (1.0 + 0.0 * visits) 

        old_q = self.Q[prev_state, prev_action]
        if terminal or next_action is None:
            td_target = prev_reward
        else:
            td_target = prev_reward + self.gamma * self.Q[next_state, next_action]

        self.Q[prev_state, prev_action] += alpha_t * (td_target - old_q)

    def reset_episode_memory(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = None

def vertex_index_from_pixel(board, vertex_pixel):
    # map pixel coordinate object -> index
    for idx, pix in board.vertex_index_to_pixel_dict.items():
        if pix == vertex_pixel:
            return idx
    raise KeyError("vertex pixel not found in vertex_index_to_pixel_dict")

def legal_indices_for_player(board, player_obj):
    # returns a list of vertex indices that are legal for the player's setup settlement
    sdict = board.get_setup_settlements(player_obj)
    legal = []
    for v_pixel in sdict.keys():
        for ii, pix in board.vertex_index_to_pixel_dict.items():
            if pix == v_pixel:
                legal.append(ii)
                break
    return legal

def bucket_value(x, boundaries):
    for i, b in enumerate(boundaries):
        if x <= b:
            return i
    return len(boundaries)

def compute_vertex_expected_value(board, v_index):
    v_pix = board.vertex_index_to_pixel_dict[v_index]
    exp = board.get_vertex_expected_resources(v_pix)
    return sum(exp.values())

def encode_agent_state(board, player_obj, phase):
    phase = int(phase)

    try:
        first_settlements = player_obj.buildGraph.get('SETTLEMENTS', [])
        if len(first_settlements) >= 1:
            first = first_settlements[0]
            if hasattr(first, 'vertexIndex'):
                first_idx = first.vertexIndex
            else:
                first_idx = vertex_index_from_pixel(board, first)
            val = compute_vertex_expected_value(board, first_idx)
        else:
            val = 0.0
    except Exception:
        val = 0.0

    my_bucket = bucket_value(val, boundaries=[1.5, 3.5]) 
    opp_count = 0
    for v in board.boardGraph.keys():
        owner = board.boardGraph[v].state.get('Player', None)
        if owner is not None and owner != player_obj:
            opp_count += 1
    if opp_count == 0:
        opp_bucket = 0
    elif opp_count <= 3:
        opp_bucket = 1
    else:
        opp_bucket = 2

    state_id = phase + 2 * (my_bucket + 3 * opp_bucket)
    return int(state_id)

def place_best_road_for_player(board, ply):
    roads = board.get_setup_roads(ply)
    if not roads:
        return None
    best = None
    best_score = -1e9
    for (v1, v2) in roads.keys():
        score = board.evaluate_road_reward((v1, v2))
        if score > best_score:
            best_score = score
            best = (v1, v2)
    if best is not None:
        ply.build_road(best[0], best[1], board)
    return best

def train_multiagent_setup(num_players=4, num_episodes=2000, alpha=0.1, gamma=0.95, eps_start=0.25, eps_end=0.05):
    template_board = catanBoard()
    n_actions = len(template_board.vertex_index_to_pixel_dict)
    n_states = 18  
    agents = [SARSAAgentPlayer(n_actions=n_actions, n_states=n_states, alpha=alpha, gamma=gamma, eps=eps_start)
              for _ in range(num_players)]

    for ep in range(1, num_episodes + 1):
        frac = (ep - 1) / max(1, num_episodes - 1)
        eps = eps_start * (1 - frac) + eps_end * frac
        for ag in agents:
            ag.epsilon = eps
            ag.reset_episode_memory()

        board = copy.deepcopy(template_board)
        colors = ['black', 'darkslateblue', 'magenta4', 'orange1']
        players = [player(chr(65 + i), colors[i % len(colors)]) for i in range(num_players)]

        forward = list(range(num_players))
        reverse = list(reversed(forward))
        turn_sequence = forward + reverse

        for turn_ptr, cur_idx in enumerate(turn_sequence):
            cur_player = players[cur_idx]
            cur_agent = agents[cur_idx]
            phase = 0 if len(cur_player.buildGraph['SETTLEMENTS']) == 0 else 1

            legal = legal_indices_for_player(board, cur_player)
            if not legal:
                if cur_agent.last_state is not None:
                    cur_agent.update_sarsa(cur_agent.last_state, cur_agent.last_action,
                                           cur_agent.last_reward, None, None, terminal=True)
                    cur_agent.last_state = None
                continue

            # compute agent's current state id (based on board and player's own first settlement if present)
            state_id = encode_agent_state(board, cur_player, phase)

            # choose action (a) for this agent now
            action = cur_agent.choose_action(state_id, legal)

            # map action index -> vertex pixel coord
            v_pix = board.vertex_index_to_pixel_dict[action]

            # execute build settlement for the acting player (this changes the board for others)
            cur_player.build_settlement(v_pix, board)

            # build best road (if any)
            place_best_road_for_player(board, cur_player)

            # compute reward for the settlement just placed for this player
            try:
                reward = board.evaluate_settlement_reward(v_pix, cur_player)
            except Exception:
                reward = 0.0

            # Choose next_state (after action). The next state is what agent would observe next.
            next_phase = 0 if len(cur_player.buildGraph['SETTLEMENTS']) == 0 else 1
            next_state_id = encode_agent_state(board, cur_player, next_phase)

            # choose a' (the next action) using current policy *at next_state* among legal actions
            # Note: legal actions at next state depend on the board immediately after other players act,
            # but SARSA uses the next action chosen under the policy. We choose a' with the current info.
            next_legal = legal_indices_for_player(board, cur_player)
            if next_legal:
                next_action = cur_agent.choose_action(next_state_id, next_legal)
            else:
                next_action = None

            # SARSA update for this agent: if there is a stored last (s,a,r) from its previous step,
            # update it now using the newly chosen (s', a').
            if cur_agent.last_state is not None:
                cur_agent.update_sarsa(cur_agent.last_state, cur_agent.last_action,
                                       cur_agent.last_reward, state_id, action, terminal=False)

            # now store the current step as last for next time the agent acts
            cur_agent.last_state = state_id
            cur_agent.last_action = action
            cur_agent.last_reward = reward
        # End of episode: finalize pending updates for each agent (their last action has no next action)
        for ag in agents:
            if ag.last_state is not None:
                ag.update_sarsa(ag.last_state, ag.last_action, ag.last_reward, None, None, terminal=True)
                ag.last_state = None
                ag.last_action = None
                ag.last_reward = None

        # optional progress
        if ep % max(1, num_episodes // 10) == 0:
            print(f"[Train] Episode {ep}/{num_episodes}  eps={eps:.4f}")

    print("[Train] Training completed.")
    return template_board, agents

# -------------------------
# Deterministic rollout using greedy policies -> final board output
# -------------------------
def deterministic_final_board(template_board, agents, num_players=4):
    # create board copy
    board = copy.deepcopy(template_board)
    colors = ['black', 'darkslateblue', 'magenta4', 'orange1']
    players = [player(chr(65 + i), colors[i % len(colors)]) for i in range(num_players)]

    forward = list(range(num_players))
    reverse = list(reversed(forward))
    turn_sequence = forward + reverse

    for cur_idx in turn_sequence:
        cur_player = players[cur_idx]
        agent = agents[cur_idx]
        phase = 0 if len(cur_player.buildGraph['SETTLEMENTS']) == 0 else 1

        legal = legal_indices_for_player(board, cur_player)
        if not legal:
            continue

        # pick best legal action according to Q
        state_id = encode_agent_state(board, cur_player, phase)
        # sort actions by Q value (high -> low), but consider only legal actions
        order = sorted(legal, key=lambda a: agent.Q[state_id, a], reverse=True)
        chosen = order[0]
        v_pix = board.vertex_index_to_pixel_dict[chosen]
        cur_player.build_settlement(v_pix, board)
        place_best_road_for_player(board, cur_player)

    # Print final chosen settlement indices per player for convenience
    for i, p in enumerate(players):
        chosen_idx = []
        for s in p.buildGraph['SETTLEMENTS']:
            # s may be pixel object or index; attempt to map to index
            try:
                if hasattr(s, 'vertexIndex'):
                    chosen_idx.append(s.vertexIndex)
                else:
                    chosen_idx.append(vertex_index_from_pixel(board, s))
            except Exception:
                chosen_idx.append(str(s))
        print(f"Player {p.color} settled at indices: {chosen_idx}")

    # try to save a screenshot of final board if catanGameView available (like in catanGame)
    if PYGAME_AVAILABLE:
        try:
            class FakeQueue:
                def __init__(self, items):
                    self.queue = items  # match what gameView expects

            class FakeGame:
                def __init__(self, board, players):
                    self.board = board
                    self.playerQueue = FakeQueue(players)
                    self.current_player = players[0] if players else None
                    self.turn = 0

            fake_game = FakeGame(board, players)
            gv = catanGameView(board, fake_game)
            gv.displayGameScreen()

            import pygame, os
            fname = os.path.join(os.getcwd(), "multiagent_sarsa_setup.png")
            surf = pygame.display.get_surface()
            if surf is not None:
                pygame.image.save(surf, fname)
                print(f"Saved final setup screenshot: {fname}")

        except Exception as e:
            print("Saving screenshot failed:", e)

    return board

# -------------------------
# Main entrypoint
# -------------------------
if __name__ == "__main__":
    NUM_PLAYERS = 4
    NUM_EPISODES = 3000    # tune as needed
    ALPHA = 0.15
    GAMMA = 0.95
    EPS_START = 0.25
    EPS_END = 0.02

    board_template, trained_agents = train_multiagent_setup(
        num_players=NUM_PLAYERS,
        num_episodes=NUM_EPISODES,
        alpha=ALPHA,
        gamma=GAMMA,
        eps_start=EPS_START,
        eps_end=EPS_END,
    )

    final_board = deterministic_final_board(board_template, trained_agents, num_players=NUM_PLAYERS)
    print("Done.")
