# test_expert.py  （调试版：更健壮地加载并打印 actor 输出用于定位问题）
import os
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import glob
import pprint

EXPERTS_DIR = "experts"

# 要测试的任务 -> 模型文件名
TASKS = {
    #"CartPole-v1": "CartPole-v1_best_expert.cleanrl_model",
    #"Acrobot-v1": "Acrobot-v1_best_expert.cleanrl_model",
    #"MountainCar-v0": "MountainCar-v0_best_expert.cleanrl_model",
    "MountainCarContinuous-v0": "MountainCarContinuous-v0_best_expert.cleanrl_model",
    "Pendulum-v1": "Pendulum-v1_best_expert.cleanrl_model",
    # 可选地添加离散任务，例如 CartPole-v1 ...
}

# ---------------- model classes (match training code) ----------------
class DQN_MLP(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 120), nn.ReLU(),
            nn.Linear(120, 84), nn.ReLU(),
            nn.Linear(84, act_dim)
        )

    def forward(self, x):
        return self.network(x)

class PPO_Agent(nn.Module):
    def __init__(self, obs_space, act_space):
        super().__init__()
        obs_dim = int(np.prod(obs_space.shape))
        act_dim = int(np.prod(act_space.shape))
        def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
            torch.nn.init.orthogonal_(layer.weight, std)
            torch.nn.init.constant_(layer.bias, bias_const)
            return layer

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        dist = torch.distributions.Normal(action_mean, action_std)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action).sum(1)
        entropy = dist.entropy().sum(1)
        value = self.critic(x)
        return action, logprob, entropy, value

class SAC_Actor(nn.Module):
    def __init__(self, obs_space, act_space):
        super().__init__()
        obs_dim = int(np.prod(obs_space.shape))
        act_dim = int(np.prod(act_space.shape))
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, act_dim)
        self.fc_logstd = nn.Linear(256, act_dim)
        action_scale = (act_space.high - act_space.low) / 2.0
        action_bias = (act_space.high + act_space.low) / 2.0
        self.register_buffer("action_scale", torch.tensor(action_scale, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor(action_bias, dtype=torch.float32))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        LOG_STD_MAX = 2
        LOG_STD_MIN = -5
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self.forward(x)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t).sum(1, keepdim=True)
        return action, log_prob, torch.tanh(mean) * self.action_scale + self.action_bias

# ---------------- utilities ----------------
def load_torch(path):
    print("[Info] loading:", path)
    state = torch.load(path, map_location="cpu")
    return state

def detect_checkpoint_type(state):
    if isinstance(state, dict):
        keys = list(state.keys())
        # PPO agent likely has keys like 'critic.0.weight', 'actor_mean.0.weight', 'actor_logstd'
        if any(k.startswith("critic") or k.startswith("actor_mean") or k == "actor_logstd" for k in keys):
            return "ppo_agent"
        # SAC bundle: contains 'actor' and 'qf1' etc
        if "actor" in state and ("qf1" in state or "qf2" in state):
            return "sac_bundle"
        # SAC actor alone: keys like 'fc1.weight' or 'fc_mean.weight' etc
        if any(k.startswith("fc1") or k.startswith("fc_mean") or k.startswith("fc_logstd") for k in keys):
            return "sac_actor"
        # DQN: keys often include 'network' or 'net' prefixes
        if any(k.startswith("network") or k.startswith("net") or k.startswith("model") for k in keys):
            return "dqn"
    return "unknown"

def try_load_state_dict(model, state_dict):
    """
    1) try direct load strict=True
    2) try strict=False
    3) try nested extraction (e.g. state['actor'] or state['agent'])
    4) try suffix-match mapping (map model keys to checkpoint keys by suffix)
    Returns (success:bool, info:str)
    """
    # direct
    try:
        model.load_state_dict(state_dict)
        return True, "loaded strict True"
    except Exception as e:
        direct_err = str(e)
    try:
        model.load_state_dict(state_dict, strict=False)
        return True, "loaded strict False"
    except Exception as e:
        loose_err = str(e)

    # nested keys
    for k in ["actor", "agent", "model", "policy", "net"]:
        if k in state_dict and isinstance(state_dict[k], dict):
            try:
                model.load_state_dict(state_dict[k], strict=False)
                return True, f"loaded nested key '{k}' (strict False)"
            except Exception:
                pass

    # suffix mapping: map model.key -> ckpt.key if ckpt.key endswith model.key
    mapped = {}
    ck_keys = list(state_dict.keys())
    model_keys = list(model.state_dict().keys())
    for mkey in model_keys:
        found = None
        for ck in ck_keys:
            if ck.endswith(mkey):
                found = ck
                break
        if found:
            mapped[mkey] = state_dict[found]
    if mapped:
        try:
            model.load_state_dict(mapped, strict=False)
            missing = set(model_keys) - set(mapped.keys())
            return True, f"loaded by suffix-mapping, missing {len(missing)} keys"
        except Exception as e:
            return False, f"suffix-mapping failed: {e}"

    return False, f"direct_err: {direct_err}; loose_err: {loose_err}"

# ---------------- env wrapper maker for PPO-like preprocessing ----------------
def make_continuous_test_env(env_id, apply_ppo_wrappers=False):
    # create base env (no vector)
    try:
        env = gym.make(env_id, render_mode="human")
    except TypeError:
        env = gym.make(env_id)
    if apply_ppo_wrappers:
        # follow ppo_continuous_action.py's make_env wrappers (single-env)
        try:
            env = gym.wrappers.FlattenObservation(env)
        except Exception:
            pass
        try:
            env = gym.wrappers.ClipAction(env)
        except Exception:
            pass
        try:
            env = gym.wrappers.NormalizeObservation(env)
        except Exception:
            pass
        try:
            env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        except Exception:
            pass
        # NormalizeReward/TransformReward not necessary for inference
    return env

# ---------------- main test routine ----------------
def debug_and_test(env_id, model_file, episodes=3, print_debug=True):
    model_path = os.path.join(EXPERTS_DIR, model_file)
    if not os.path.exists(model_path):
        print("[Skip] model not found:", model_path)
        return

    state = load_torch(model_path)
    ck_type = detect_checkpoint_type(state)
    print("[Info] Detected checkpoint type:", ck_type)
    # create env; for PPO-type checkpoints we replicate the same wrappers that training used
    apply_ppo_wrappers = (ck_type == "ppo_agent")
    env = make_continuous_test_env(env_id, apply_ppo_wrappers=apply_ppo_wrappers)
    obs_space = env.observation_space
    act_space = env.action_space

    policy = None
    load_info = "not attempted"
    if ck_type == "ppo_agent":
        policy = PPO_Agent(obs_space, act_space)
        load_ok, load_info = try_load_state_dict(policy, state if isinstance(state, dict) else {})
    elif ck_type == "sac_bundle":
        # try to extract actor first
        actor_state = state.get("actor") or state.get("actor_state_dict") or state.get("policy") or None
        if actor_state is None:
            # fallback: maybe checkpoint has top-level actor keys (fc1, fc_mean ...)
            actor_state = {k: v for k, v in state.items() if any(prefix in k for prefix in ("fc1", "fc2", "fc_mean", "fc_logstd", "action_scale", "action_bias"))}
        policy = SAC_Actor(obs_space, act_space)
        load_ok, load_info = try_load_state_dict(policy, actor_state if isinstance(actor_state, dict) else {})
    elif ck_type == "sac_actor":
        policy = SAC_Actor(obs_space, act_space)
        load_ok, load_info = try_load_state_dict(policy, state)
    elif ck_type == "dqn":
        obs_dim = int(np.prod(obs_space.shape))
        act_dim = act_space.n if isinstance(act_space, gym.spaces.Discrete) else int(np.prod(act_space.shape))
        policy = DQN_MLP(obs_dim, act_dim)
        load_ok, load_info = try_load_state_dict(policy, state)
    else:
        # try detect by keys
        if isinstance(state, dict) and any(k.startswith("actor_mean") for k in state.keys()):
            policy = PPO_Agent(obs_space, act_space)
            load_ok, load_info = try_load_state_dict(policy, state)
            ck_type = "ppo_agent"
        else:
            print("[Error] Unknown checkpoint format. Keys:", list(state.keys())[:20])
            return

    print("[Info] Load result:", load_info)
    # after load, print some parameter/statistics to debug whether actor is sane
    policy.eval()

    # print param counts and a few param norms
    total_params = sum(p.numel() for p in policy.parameters())
    print(f"[Info] policy params: {total_params}")
    # print first layer mean/std
    first_params = list(policy.parameters())[:4]
    for i, p in enumerate(first_params):
        print(f"param[{i}] shape={tuple(p.shape)}, mean={p.data.mean():.6f}, std={p.data.std():.6f}")

    # run episodes and at each reset print actor mean/logstd/sample for the initial obs
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        step = 0
        # sample debug on initial observation
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            if hasattr(policy, "actor_mean") or hasattr(policy, "actor_logstd"):
                # PPO agent
                if hasattr(policy, "actor_mean"):
                    mean = policy.actor_mean(obs_t)
                else:
                    mean = None
                logstd = getattr(policy, "actor_logstd", None)
                if logstd is not None:
                    std = torch.exp(logstd)
                else:
                    std = None
                print(f"[Debug ep{ep}] initial mean (shape) = {None if mean is None else mean.cpu().numpy().tolist()}")
                print(f"[Debug ep{ep}] actor_logstd (shape) = {None if std is None else std.cpu().numpy().tolist()}")
            elif hasattr(policy, "forward") and isinstance(policy, SAC_Actor):
                mean, log_std = policy.forward(obs_t)
                std = torch.exp(log_std)
                print(f"[Debug ep{ep}] SAC actor mean = {mean.cpu().numpy().tolist()}")
                print(f"[Debug ep{ep}] SAC actor std = {std.cpu().numpy().tolist()}")

        while not (done or truncated):
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                if isinstance(act_space, gym.spaces.Discrete):
                    q = policy(obs_t)
                    action = int(torch.argmax(q, dim=1).item())
                else:
                    if ck_type == "ppo_agent":
                        a, _, _, _ = policy.get_action_and_value(obs_t)
                        action = a.cpu().numpy().squeeze(0)
                    else:  # sac
                        a, _, _ = policy.get_action(obs_t)
                        action = a.cpu().numpy().squeeze(0)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += float(reward)
            step += 1
            if step > 5000:
                print("[Warning] too many steps, breaking")
                break
        print(f"Episode {ep+1}: total_reward={total_reward:.3f}, steps={step}")
    try:
        env.close()
    except Exception:
        pass

# ---------------- run tests ----------------
if __name__ == "__main__":
    for env_id, fname in TASKS.items():
        path = os.path.join(EXPERTS_DIR, fname)
        if not os.path.exists(path):
            print("[Skip] missing", path)
            continue
        print("="*60)
        print("Testing", env_id, fname)
        debug_and_test(env_id, fname, episodes=3, print_debug=True)
