import gym_carla
import gym
import numpy as np
from src.carla_gym.controllers.barc_pid import PIDWrapper  # same expert used in il_CAD_BARC_trainer

def run_pid_in_barc(max_laps: int = 500):
    """
    Run the PID expert inside the barc-v0 environment until the desired number of laps are completed.
    
    Args:
        max_laps (int): number of laps to complete before stopping
    """

    # -------------------------------
    # 1. Hardcoded CARLA and env setup
    # -------------------------------
    carla_params = dict(
        dt=0.1,
        t0=0.0,
        dt_sim=0.01,
        max_n_laps=max_laps,
        do_render=True,
        enable_camera=False, # no need to have carla visualization here
        host='localhost',
        port=2000,
        weatherID=0,
        map_name='/Game/L_track_barc1/Maps/L_track_barc1/L_track_barc1',
        track_name='L_track_barc',
    )

    print("Initializing barc-v0 environment...")
    env = gym.make('barc-v0', **carla_params)

    # -------------------------------
    # 2. Initialize PID controller
    # -------------------------------
    pid_controller = PIDWrapper(
        dt=carla_params["dt"],
        t0=carla_params["t0"],
        track_obj=env.get_track(),
    )

    # -------------------------------
    # 3. Reset environment
    # -------------------------------
    obs, info = env.reset(
        options={"controller": pid_controller, 'spawning' : 'fixed'},
        map_name=carla_params["map_name"],
        weatherID=carla_params["weatherID"],
    )
    pid_controller.reset(options=info)

    total_reward = 0.0
    completed_laps = 0
    step_count = 0

    # -------------------------------
    # 4. Main control loop
    # -------------------------------
    while completed_laps < max_laps:
        action, ctrl_info = pid_controller.step(**obs, **info)
        action = np.clip(action, env.action_space.low, env.action_space.high)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        step_count += 1

        if terminated:
            completed_laps = int(info.get("lap_no", completed_laps + 1))
            print(f"✅ Lap {completed_laps} completed — Lap time: {info.get('lap_time', 0):.2f}s")

        if truncated:
            print("⚠️ Simulation truncated (vehicle likely went off track). Ending run.")
            break

    # -------------------------------
    # 5. Wrap up
    # -------------------------------
    env.close()
    print(f"\nSimulation finished after {step_count} steps.")
    print(f"Total reward collected: {total_reward:.2f}")
    print(f"Laps completed: {completed_laps}/{max_laps}")
    return {"steps": step_count, "reward": total_reward, "laps": completed_laps}

if __name__ == '__main__':
    run_pid_in_barc()