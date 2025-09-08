#!/usr/bin/env python3
# train_expert.py
# 一键训练并保存 experts/ 下的专家模型（适配已安装的 cleanrl 脚本）
# 增强：在训练前检查 experts/ 是否已有模型，若存在则跳过训练

import os
import sys
import time
import re
import glob
import shutil
import subprocess

# 任务 -> cleanrl 脚本 映射（你已把这些脚本放到 cleanrl 包或 site-packages）
ALGO_MAP = {
    "CartPole-v1": "dqn.py",
    "Acrobot-v1": "dqn.py",
    "MountainCar-v0": "dqn.py",
    "MountainCarContinuous-v0": "ppo_continuous_action.py",
    "Pendulum-v1": "sac_continuous_action.py",
}

# 建议训练步数（可按需调整）
TIMESTEPS = {
    "CartPole-v1": 500_000,
    "Acrobot-v1": 500_000,
    "MountainCar-v0": 1_000_000,
    "MountainCarContinuous-v0": 500_000,
    "Pendulum-v1": 500_000,
}

EXPERTS_DIR = "experts"
os.makedirs(EXPERTS_DIR, exist_ok=True)

# 建议参数映射（仅在脚本源码包含对应字段时才传入）
SUGGESTED_PARAMS = {
    "dqn.py": {
        "--save-model": ("save_model", None),
        "--learning-rate": ("learning_rate", "0.0005"),
        "--buffer-size": ("buffer_size", "500000"),
        "--learning-starts": ("learning_starts", "1000"),
        "--train-frequency": ("train_frequency", "1"),
        "--target-network-frequency": ("target_network_frequency", "1000"),
        "--start-e": ("start_e", "1"),
        "--end-e": ("end_e", "0.01"),
        "--exploration-fraction": ("exploration_fraction", "0.8"),
        "--batch-size": ("batch_size", "64"),
    },
    "sac_continuous_action.py": {
        "--policy-lr": ("policy_lr", "3e-4"),
        "--q-lr": ("q_lr", "1e-3"),
        "--buffer-size": ("buffer_size", "1000000"),
        "--learning-starts": ("learning_starts", "5000"),
        "--batch-size": ("batch_size", "256"),
        "--policy-frequency": ("policy_frequency", "2"),
        "--target-network-frequency": ("target_network_frequency", "1"),
    },
    "ppo.py": {
        "--learning-rate": ("learning_rate", "2.5e-4"),
        "--num-envs": ("num_envs", "4"),
        "--num-steps": ("num_steps", "128"),
    },
}


def find_cleanrl_script_path(script_name):
    """查找 cleanrl/<script_name> 的路径"""
    # 1) installed cleanrl package
    try:
        import cleanrl  # type: ignore
        pkg_dir = os.path.dirname(cleanrl.__file__)
        candidate = os.path.join(pkg_dir, script_name)
        if os.path.isfile(candidate):
            return candidate
    except Exception:
        pass

    # 2) search sys.path for "cleanrl" directory
    for p in sys.path:
        candidate_dir = os.path.join(p, "cleanrl")
        if os.path.isdir(candidate_dir):
            candidate = os.path.join(candidate_dir, script_name)
            if os.path.isfile(candidate):
                return candidate

    # 3) relative ./cleanrl folder (if you cloned into project)
    candidate = os.path.join(os.getcwd(), "cleanrl", script_name)
    if os.path.isfile(candidate):
        return candidate

    return None


def script_supports_field(script_path, field_name):
    """检测脚本源码中是否出现 dataclass 字段名（例如 'save_model' 或 'policy_lr'）"""
    try:
        txt = open(script_path, "r", encoding="utf-8").read()
    except Exception:
        return False
    return field_name in txt


def find_model_in_run_dir(run_dir):
    """在 runs/run_dir 下查找可能的模型文件（返回第一个匹配或最新修改的）"""
    if not os.path.isdir(run_dir):
        return None
    pattern_list = ["**/*.cleanrl_model", "**/*.pt", "**/*.pth", "**/*.pkl", "**/*.bin"]
    files = []
    for pat in pattern_list:
        files.extend(glob.glob(os.path.join(run_dir, pat), recursive=True))
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def find_existing_expert_file(env_id):
    """在 experts/ 目录查找是否已有对应模型文件"""
    patterns = [
        f"{env_id}_best_expert.*",
        f"expert_{env_id}*",
        f"{env_id}_best_expert*",
        f"*{env_id}*cleanrl_model",
        f"{env_id}*expert*",
    ]
    for pat in patterns:
        matches = glob.glob(os.path.join(EXPERTS_DIR, pat))
        if matches:
            return matches[0]
    return None


def train_one(env_id):
    """为单个 env 训练 expert，若 experts/ 已有则跳过"""
    existing = find_existing_expert_file(env_id)
    if existing:
        print(f"[ExpertManager] Found existing expert for {env_id}: {existing}  --> skipping training.")
        return existing

    if env_id not in ALGO_MAP:
        print(f"[Error] no algorithm mapping for {env_id}")
        return None

    algo_script = ALGO_MAP[env_id]
    script_path = find_cleanrl_script_path(algo_script)
    if script_path is None:
        print(f"[Error] could not locate cleanrl script {algo_script}. Make sure cleanrl is installed or cleanrl/{algo_script} exists.")
        return None

    timesteps = TIMESTEPS.get(env_id, 200_000)
    timestamp = int(time.time())
    exp_name = f"expert_{env_id}_{timestamp}"
    run_dir_glob = f"runs/{env_id}__{exp_name}*"
    run_dir = f"runs/{env_id}__{exp_name}"

    cmd = [sys.executable, script_path, "--env-id", env_id, "--total-timesteps", str(timesteps), "--exp-name", exp_name]

    suggested = SUGGESTED_PARAMS.get(algo_script, {})
    for flag, (field_name, value) in suggested.items():
        if script_supports_field(script_path, field_name):
            if value is None:
                cmd.append(flag)
            else:
                cmd.extend([flag, str(value)])

    print(f"[ExpertManager] Running training for {env_id} with script {script_path}")
    print("COMMAND:", " ".join(cmd))

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    best_reward = -float("inf")
    best_model_dst_base = os.path.join(EXPERTS_DIR, f"{env_id}_best_expert")  # will add extension

    try:
        for line in proc.stdout:
            print(line, end="")

            m = re.search(r"episodic_return=\[([^\]]+)\]", line)
            if m:
                try:
                    val = float(m.group(1).strip().strip(","))
                except Exception:
                    continue

                # === 新增过滤机制：Pendulum-v1 时，忽略 reward > -70 的异常好值 ===
                if env_id == "Pendulum-v1" and val > -70:
                    print(f"[Filter] Skip episodic_return={val:.2f} (> -70, ignored)")
                    continue

                if val > best_reward:
                    best_reward = val
                    matches = glob.glob(run_dir_glob)
                    model_found = None
                    for d in matches:
                        cand = find_model_in_run_dir(d)
                        if cand:
                            model_found = cand
                            break
                    if model_found:
                        _, ext = os.path.splitext(model_found)
                        dst = best_model_dst_base + ext
                        try:
                            shutil.copyfile(model_found, dst)
                            print(f"[ExpertManager] New best model saved to {dst} (reward={best_reward:.2f})")
                        except Exception as e:
                            print("[ExpertManager] copy failed:", e)

        proc.wait()
    except KeyboardInterrupt:
        print("[ExpertManager] Interrupted by user, terminating child process...")
        proc.terminate()
        proc.wait()

    print(f"[ExpertManager] Training finished for {env_id}, best reward={best_reward if best_reward != -float('inf') else 'N/A'}")

    matches = glob.glob(run_dir_glob)
    model_candidate = None
    for d in matches:
        m = find_model_in_run_dir(d)
        if m:
            model_candidate = m
            break

    if model_candidate:
        _, ext = os.path.splitext(model_candidate)
        dst = best_model_dst_base + ext
        try:
            shutil.copyfile(model_candidate, dst)
            print(f"[ExpertManager] Copied model from run dir to {dst}")
            return dst
        except Exception as e:
            print("[ExpertManager] Final copy failed:", e)
            return None
    else:
        print("[Warning] No model file found in run directories for", env_id)
        return None


def main():
    tasks = ["CartPole-v1", "Acrobot-v1", "MountainCar-v0", "MountainCarContinuous-v0", "Pendulum-v1"]
    for env in tasks:
        print("=" * 60)
        print("Training expert for", env)
        model = train_one(env)
        if model:
            print(f"[Main] Expert ready for {env}: {model}")
        else:
            print(f"[Main] Expert training for {env} failed or no model saved.")


if __name__ == "__main__":
    main()
