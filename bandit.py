import json
import numpy as np
import matplotlib.pyplot as plt
from simulator import score, image_to_vector, persona_to_vector, get_feature_keys

with open("image_features.json") as f:
    image_features = json.load(f)

with open("personas.json") as f:
    personas = json.load(f)["personas"]

# filter out any images that failed to parse
images = {k: v for k, v in image_features.items() if "_parse_error" not in v}
image_names = sorted(images.keys())
n_arms = len(image_names)
n_features = len(get_feature_keys())

# precompute feature vectors for all images
image_vectors = np.array([image_to_vector(images[name]) for name in image_names])

N_ROUNDS = 500
N_RUNS = 20


def oracle_best_score(persona, rng):
    """Compute the best possible expected score (no noise) for regret calculation."""
    scores = []
    for name in image_names:
        avg = np.mean([score(persona, images[name], rng) for _ in range(200)])
        scores.append(avg)
    return max(scores)


# --- Policies ---

class RandomPolicy:
    def __init__(self, n_arms, n_features):
        pass

    def select_arm(self, context, rng):
        return rng.integers(n_arms)

    def update(self, arm, context, reward):
        pass


class GreedyPolicy:
    def __init__(self, n_arms, n_features):
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select_arm(self, context, rng):
        unexplored = np.where(self.counts == 0)[0]
        if len(unexplored) > 0:
            return rng.choice(unexplored)
        return int(np.argmax(self.values))

    def update(self, arm, context, reward):
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]


class EpsilonGreedyPolicy:
    def __init__(self, n_arms, n_features, epsilon=0.1):
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select_arm(self, context, rng):
        unexplored = np.where(self.counts == 0)[0]
        if len(unexplored) > 0:
            return rng.choice(unexplored)
        if rng.random() < self.epsilon:
            return rng.integers(n_arms)
        return int(np.argmax(self.values))

    def update(self, arm, context, reward):
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]


class LinUCBPolicy:
    def __init__(self, n_arms, n_features, alpha=1.0):
        self.alpha = alpha
        self.A = [np.eye(n_features) for _ in range(n_arms)]
        self.b = [np.zeros(n_features) for _ in range(n_arms)]

    def select_arm(self, context, rng):
        ucbs = []
        for a in range(n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            ucb = context @ theta + self.alpha * np.sqrt(context @ A_inv @ context)
            ucbs.append(ucb)
        return int(np.argmax(ucbs))

    def update(self, arm, context, reward):
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context


POLICIES = {
    "Random": RandomPolicy,
    "Greedy": GreedyPolicy,
    "Epsilon-Greedy": EpsilonGreedyPolicy,
    "LinUCB": LinUCBPolicy,
}


def run_experiment(persona, policy_class, n_rounds, rng):
    policy = policy_class(n_arms, n_features)
    rewards = []
    for _ in range(n_rounds):
        context = persona_to_vector(persona)
        arm = policy.select_arm(context, rng)
        reward = score(persona, images[image_names[arm]], rng)
        policy.update(arm, context, reward)
        rewards.append(reward)
    return np.array(rewards)


if __name__ == "__main__":
    fig, axes = plt.subplots(len(personas), 1, figsize=(10, 4 * len(personas)))

    for idx, persona in enumerate(personas):
        ax = axes[idx]
        print(f"\nRunning experiments for {persona['name']}...")

        rng = np.random.default_rng(42)
        best = oracle_best_score(persona, rng)
        print(f"  Oracle best expected score: {best:.3f}")

        for policy_name, policy_class in POLICIES.items():
            all_regrets = []
            for run in range(N_RUNS):
                rng = np.random.default_rng(run)
                rewards = run_experiment(persona, policy_class, N_ROUNDS, rng)
                regret = np.cumsum(best - rewards)
                all_regrets.append(regret)

            mean_regret = np.mean(all_regrets, axis=0)
            ax.plot(mean_regret, label=policy_name)

        ax.set_title(f"{persona['name']} — Cumulative Regret")
        ax.set_xlabel("Round")
        ax.set_ylabel("Cumulative Regret")
        ax.legend()

    plt.tight_layout()
    plt.savefig("regret_curves.png", dpi=150)
    plt.show()
    print("\nSaved regret_curves.png")
