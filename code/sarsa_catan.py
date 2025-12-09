from typing import List, Optional, Iterable, Tuple
import numpy as np
import random

class SARSASetupAgent:
    def __init__(self, env, alpha: float = 0.1, gamma: float = 0.99, epsilon: float = 0.2, phases: int = 2):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.phases = phases
        self.n_actions = int(self.env.get_num_actions())
        self.Q = np.zeros((self.phases, self.n_actions), dtype=float)
        self.visit_counts = np.zeros((self.phases, self.n_actions), dtype=int)
        self.alpha0 = float(self.alpha)


    def choose_action(self, phase: int, legal_actions: Iterable[int]):
        legal = list(legal_actions)
        if not legal:
            raise ValueError("No legal actions available")
        if random.random() < self.epsilon:
            return random.choice(legal)

        qvals = self.Q[phase]
        best_val = max(qvals[a] for a in legal)
        best_actions = [a for a in legal if qvals[a] == best_val]
        return int(random.choice(best_actions))

    def train(self, num_episodes=1000, max_steps_per_episode=100, eps_start=0.5, eps_end=0.01, alpha0: float = None, alpha_decay: float = 0.0):
        if alpha0 is None:
            alpha0 = float(self.alpha0)
        self.epsilon = float(eps_start)
        for ep in range(1, num_episodes + 1):
            frac = (ep - 1) / max(1, num_episodes - 1)
            self.epsilon = float(eps_start) * (1 - frac) + float(eps_end) * frac
            state = self.env.reset()
            phase = 0

            legal = self.env.get_legal_actions()
            a = self.choose_action(phase, legal)

            total_reward = 0.0
            step = 0
            done = False

            while not done and step < max_steps_per_episode:
                next_state, reward, done, info = self.env.step(a)

                if isinstance(info, dict) and 'phase' in info:
                    next_phase = int(info['phase'])
                else:
                    next_phase = min(self.phases - 1, phase + 1)
                try:
                    self.visit_counts[phase, a] += 1
                except Exception:
                    pass

                visits = int(self.visit_counts[phase, a]) if a is not None else 0
                if alpha_decay and alpha_decay > 0:
                    alpha_t = alpha0 / (1.0 + alpha_decay * visits)
                else:
                    alpha_t = alpha0 / (1.0 + visits)

                if not done:
                    next_legal = self.env.get_legal_actions()
                    a_next = self.choose_action(next_phase, next_legal)
                    td_target = reward + self.gamma * self.Q[next_phase, a_next]
                else:
                    a_next = None
                    td_target = reward

                old_q = self.Q[phase, a]
                # perform SARSA update with adaptive alpha
                self.Q[phase, a] += alpha_t * (td_target - old_q)
                total_reward += reward
                state = next_state
                phase = next_phase
                a = a_next if a_next is not None else 0
                step += 1

            if (ep % max(1, num_episodes // 10) == 0):
                print(f"Episode {ep}/{num_episodes})")
        print(self.Q)
        print(self.visit_counts)
        return self.Q

    