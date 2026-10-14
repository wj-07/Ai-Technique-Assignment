import json
import pickle
import socket
import random
import numpy as np

QTABLE_PATH = "scene4_qtable.pkl"
EPISODES_TO_SHOW = 100

def state_to_key(state_dict):
    return (
        state_dict["px"],
        state_dict["py"],
        state_dict["floor"],
        state_dict["enemy"],
        state_dict["apple_dir"],
        state_dict.get("apple_count", 0),
    )

def choose_action(q_table, state_key, epsilon=0.05):
    # 5% reflex randomness to handle physics variations
    if random.random() < epsilon:
        return random.randint(0, 5)
        
    if state_key not in q_table:
        return random.randint(0, 5)
        
    q_vals = q_table[state_key]
    return int(np.argmax(q_vals))

def main():
    with open(QTABLE_PATH, "rb") as f:
        q_table = pickle.load(f)
    print(f"[Demo] Loaded trained Q-table: {len(q_table)} known states")

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(("127.0.0.1", 9999))
    server_socket.listen(1)
    
    print("==================================================")
    print("[Demo] Waiting for Godot... (RECORD AT 8.0x SPEED)")
    print("==================================================")

    client_socket, addr = server_socket.accept()
    print(f"[Demo] Godot connected from {addr}\n")

    episode = 0
    current_episode_reward = 0.0
    apples_history = []
    buffer = ""

    while episode < EPISODES_TO_SHOW:
        data = client_socket.recv(4096).decode("utf-8")
        if not data:
            break

        buffer += data
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            if not line.strip():
                continue

            payload = json.loads(line)
            state_key = state_to_key(payload["state"])
            reward = payload["reward"]
            done = payload["done"]
            apples = payload.get("apples", 0)
            distance = payload.get("distance", 0.0)

            current_episode_reward += reward
            action = choose_action(q_table, state_key)

            if done:
                episode += 1
                apples_history.append(apples)
                result = "SUCCESS (2/2 apples!)" if apples >= 2 else "ended"
                
                print(f"[Demo] Episode {episode:04d}/{EPISODES_TO_SHOW} | Total Reward: {current_episode_reward:7.2f} | Apples: {apples} | Dist: {distance:6.1f}px | {result}")
                
                current_episode_reward = 0.0

            response = json.dumps({"action": action}) + "\n"
            client_socket.sendall(response.encode("utf-8"))

    client_socket.close()
    server_socket.close()
    
    # --- Final Demo Success Rate Summary ---
    if len(apples_history) > 0:
        successful_episodes = sum(1 for a in apples_history if a >= 2)
        overall_success_rate = (successful_episodes / len(apples_history)) * 100.0
        
        print("\n=== Success Rate (2/2 Apples Collected) ===")
        print(f"Overall ({len(apples_history)} episodes) : {overall_success_rate:.1f}%")
        print("===========================================")

    print("\n[Demo] Playback finished.")

if __name__ == "__main__":
    main()