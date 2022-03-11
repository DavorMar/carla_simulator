import sys
from carla_environment import ENV

import carla
import time
import numpy as np
from ddpg_tf2 import Agent
from utils import plotLearning

#Global variables(epochs and frames per epoch, 10frames = 1 second in environment)
RUNS = 3000
FRAMES = 100
epsilon = 1
EPSILON_DECAY = 0.9995 ## 0.9975 99975 #0.95
MIN_EPSILON = 0.001

if __name__ == "__main__":
    #Generate environment , define env settings, check for loading trained model
    env = ENV()
    settings = env.world.get_settings()
    original_settings = env.world.get_settings()
    settings.synchronous_mode = True  # Enables synchronous mode
    settings.fixed_delta_seconds = 0.1#1 frame = 0.1 second

    env.world.apply_settings(settings)
    figure_file = "plots\\autopilot_png"
    best_score = -1000#to do: create a saved best score and load it
    agent = Agent(input_dims=env.observation_space_shape, env=env,
                  n_actions=1)
    score_history = []
    #train should be False if you just want to test the model
    train = True
    #check if we are loading a checkpoint
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
        # evaluate = False
    else:
        # evaluate = True
        print("TRAINING DEACTIVATED")

    #load in memory and model(THIS IS REQUIRED, OTHERWISE TENSORFLOW CANT JUST LOAD THE MODEL)
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
        #Tensorflow requires us to do these steps in order to load model
        while n_steps < len(state_memory):
            action = env.sample_action()
            agent.remember(state_memory[n_steps],action_memory[n_steps],reward_memory[n_steps],new_state_memory[n_steps],terminal_memory[n_steps])
            n_steps+=1

        agent.learn()
        agent.load_models()
    else:
        print("NOT LOADING CHECKPOINT")
    time.sleep(2)


    total_frames = 0
    #RUNS or EPOCHS
    for i in range(RUNS):
        try:
            env.reset()
            # spawn = True is used so first points collected dont generate a reward
            _, _, _, observation = env.step(env.route_points,spawn=True)
            done = False
            score = 0
            frame = 0
            #Done means if there was a collision
            while not done and frame < FRAMES:
                env.world.tick()
                if np.random.random() < epsilon and train == True:#exploit vs explore
                    action = env.sample_action()
                    action_type = "Random"
                else:
                    action_type = "Agent"

                    action = agent.choose_action(observation)


                _, reward, done, observation_ = env.step(env.route_points,action)

                score += reward
                sys.stdout.write(
                    f"\rFrame: {frame}/{FRAMES}, Action: Steer {format(float(action[0]), '.2f')} ,Velocity and distance:{observation[-3]},{round(observation[-1], 2)},Action is {action_type} {format(float(epsilon), '.2f')}, Reward is {reward}" + "          " + "\n")
                sys.stdout.flush()

                agent.remember(observation, action, reward, observation_, done)
                #if not training, but just testing, agent should do gradient descent
                if train:
                    agent.learn()
                observation = observation_
                frame +=1
                total_frames += 1

            score_history.append(score)
            avg_score = np.mean(score_history[-50:])
            if avg_score > best_score and i > 50:
                best_score = avg_score
                if train:
                    agent.save_memory()
                    agent.save_models()

            print('episode ', i, 'score %.1f' % score, 'avg score %.1f' % avg_score, "best score %.1f" % best_score,end = "\r")
        #if some sensor is not loaded or connection is lost to host(happens alot actually)
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

        #destroy all actors at the end of each run/epoch
        finally:

            if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY
                epsilon = max(MIN_EPSILON, epsilon)
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
