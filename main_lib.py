import numpy as np
import copy


def policy_iteration(env, gamma=0.99, theta=1e-8):
    V = np.zeros(env.nS)
    policy = np.ones([env.nS, env.nA]) / env.nA

    while True:
        V = policy_evaluation(env, policy, gamma, theta)
        new_policy = policy_improvement(env, V)

        if (new_policy == policy).all():
            break

        policy = copy.copy(new_policy)

    return policy, V


def value_iteration(env, gamma=0.99, theta=1e-8):
    V = np.zeros(env.nS)
    policy = np.ones([env.nS, env.nA]) / env.nA

    while True:
        delta, V = value_evaluation(env, V, gamma)

        if delta < theta:
            break

    policy = policy_improvement(env, V, gamma)

    return policy, V


def policy_evaluation(env, policy, gamma=0.99, theta=1e-8):
    V = np.zeros(env.nS)

    while True:
        delta = 0

        for state in range(env.nS):
            Vs = 0
            for action, action_prob in enumerate(policy[state]):
                for prob, next_state, reward in env.MDP[state][action]:
                    Vs += action_prob * prob * (reward + gamma * V[next_state])
            delta = max(delta, np.abs(V[state] - Vs))
            V[state] = Vs

        if delta < theta:
            break

    return V


def policy_improvement(env, V, gamma=0.99):
    policy = np.zeros([env.nS, env.nA]) / env.nA

    for s in range(env.nS):
        mdp = v_to_mdp(env, V, s, gamma)
        a = np.argmax(mdp)

        policy[s][a] = 1

    return policy


def value_evaluation(env, V, gamma=0.99):
    delta = 0

    for state in range(env.nS):
        v = V[state]
        V[state] = max(v_to_mdp(env, V, state, gamma))
        delta = max(delta, abs(V[state] - v))

    return delta, V


def v_to_mdp(env, V, state, gamma):
    mdp = np.zeros(env.nA)

    for action in range(env.nA):
        for prob, next_state, reward in env.MDP[state][action]:
            mdp[action] += prob * (reward + gamma * V[next_state])

    return mdp
