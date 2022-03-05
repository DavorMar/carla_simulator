import sys
from carla_environment import ENV
import carla
import time
import numpy as np
from ddpg_tf2 import Agent
from utils import plotLearning


RUNS = 2000
FRAMES = 100

if __name__ == "__main__":
    env = ENV()
    settings = env.world.get_settings()
    original_settings = env.world.get_settings()
    settings.synchronous_mode = True  # Enables synchronous mode
    settings.fixed_delta_seconds = 0.1

    env.world.apply_settings(settings)
    figure_file = "plots\\autopilot_png"
    best_score = -1000
    agent = Agent(input_dims=env.observation_space_shape, env=env,
                  n_actions=env.action_space.shape[0])
    score_history = []

    train = True

    check_load = False
    while not check_load:
        load_preference = input("Do you want to load checkpoint? (Y/N): ")
        if load_preference == "Y":
            load_checkpoint = True
            check_load = True
        elif load_preference == "N":
            load_checkpoint = False
            check_load = True
        else:
            print("Please answer with Y or N")

    if train:
        print("TRAINING ACTIVATED")
        evaluate = False
    else:
        evaluate = True
        print("TRAINING DEACTIVATED")
    if load_checkpoint:
        print("LOADING CHECKPOINT")
        chkpt_dir = "tmp\\memory"
        file_name = "memory"

        n_steps = 0
        state_memory = np.load(fr"{chkpt_dir}\{file_name}-state.npy")
        new_state_memory = np.load(fr"{chkpt_dir}\{file_name}-states_.npy")
        action_memory = np.load(fr"{chkpt_dir}\{file_name}-actions.npy")
        reward_memory = np.load(fr"{chkpt_dir}\{file_name}-rewards.npy")
        terminal_memory = np.load(fr"{chkpt_dir}\{file_name}-dones.npy")
        while n_steps < len(state_memory):
            # print(state_memory.shape)
            # print(n_steps)
            action = env.sample_action()
            agent.remember(state_memory[n_steps],action_memory[n_steps],reward_memory[n_steps],new_state_memory[n_steps],terminal_memory[n_steps])
            n_steps+=1
        # agent.load_memory()

        agent.learn()
        agent.load_models()
    else:
        print("NOT LOADING CHECKPOINT")
    time.sleep(5)
    total_frames = 0
    for i in range(RUNS):
        try:
            env.reset()
            _, _, _, observation = env.step(env.route_points,spawn=True)
            done = False
            score = 0
            frame = 0
            while not done and frame < FRAMES:

                env.world.tick()
                if total_frames < agent.batch_size:
                    action = env.sample_action()
                else:
                    action = agent.choose_action(observation, evaluate)
                sys.stdout.write(
                    f"\rFrame is: {frame}/{FRAMES}, Action is: {float(action[0])}\|{float(action[1])},Velocity and distance are:{observation[-3]},{round(observation[-1],2)}"+"          " +"\n")
                sys.stdout.flush()
                _, reward, done, observation_ = env.step(env.route_points,action)

                score += reward
                agent.remember(observation, action, reward, observation_, done)
                done = done
                if train:
                    agent.learn()
                observation = observation_
                frame +=1

            score_history.append(score)
            avg_score = np.mean(score_history[-50:])

            if avg_score > best_score and i > 50:
                best_score = avg_score
                if train:
                    agent.save_memory()
                    agent.save_models()

            print('episode ', i, 'score %.1f' % score, 'avg score %.1f' % avg_score, "best score %.1f" % best_score,end = "\r")
        except (UnboundLocalError,RuntimeError):
            time.sleep(1)
            print("""
                    ###############
                    ERRORED UNBOUND
                    ###############
             """)

            pass
        except KeyboardInterrupt:
            check = False
            while check is False:
                answer = input("Do you want to save data?(Y/N):")
                if answer == "Y":
                    agent.save_memory()
                    agent.save_models()
                    check = True
                elif answer == "N":
                    check = True
                else:
                    print("Please answer with Y or N")
            break


        finally:
            print("""
                    ###################
                    ALL ACTORS DESTROYED
                    ####################
            """)
            env.client.apply_batch([carla.command.DestroyActor(x) for x in env.actor_list])
    # try:
    #     if train:
    #         x = [i + 1 for i in range(int(RUNS/10))]
    #         plotLearning(x, score_history, figure_file)
    # except:
    #     pass
