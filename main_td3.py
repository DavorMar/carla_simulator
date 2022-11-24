import gym
import numpy as np
from TD3 import model
# from utils import plot_learning_curve
from carla_parking_env import ENV

import carla
import time

if __name__ == '__main__':
    #env = gym.make('LunarLanderContinuous-v2')
    #env = gym.make('Pendulum-v0')
    env = ENV(actions_type="C")
    print(env.observation_space.shape, "OBSERVATION")
    print(env.action_space.shape[0], "OBSERVATION")
    agent = model.Agent(alpha=0.0002, beta=0.0002,
            input_dims=env.observation_space.shape, tau=0.001,
            env=env, batch_size=100, layer1_size=400, layer2_size=300,
            n_actions=env.action_space.shape[0])
    n_games = 5000
    FRAMES = 700
    filename = 'plots/' + 'walker_' + str(n_games) + '_games.png'

    best_score = 80
    avg_score = 0
    score_history = []
    answer_load_models = False
    while not answer_load_models:
        answer = input("Would you like to load models? (y/n) ")
        answer_load_models = True
        if answer.lower() == "y":
            agent.load_models()
            answer_load_models = True
            agent.warmup = 5
        elif answer.lower() == "n":
            pass
        else:
            print("Please asnwer with y or n")


    for i in range(0,n_games):
        try:
            env = ENV(actions_type="C")
            settings = env.world.get_settings()
            settings.no_rendering_mode = True
            original_settings = env.world.get_settings()
            settings.synchronous_mode = True  # Enables synchronous mode
            settings.fixed_delta_seconds = 0.1  # 1 frame = 0.1 second

            env.world.apply_settings(settings)
            env.world.unload_map_layer(carla.MapLayer.Walls)
            env.reset()
            # env.world.unload_map_layer(carla.MapLayer.Props)
            observation, _, _ = env.step([0,0])
            env.world.tick()
            done = False
            score = 0
            for x in range(FRAMES):
                start_time = time.time()
                action = agent.choose_action(observation)
                # print(action , "YYYYYYYYYYY")
                observation_, reward, done = env.step(action)
                if x == FRAMES - 1:
                    done = True

                agent.remember(observation, action, reward, observation_, done)
                agent.learn()
                score += reward
                observation = observation_
                env.world.tick()
                if done:
                    break
                print("FPS: ", int(1.0 / (time.time() - start_time)),
                      np.around([float(action[0]),float(action[1])],2))
                # print(np.around(observation_, 4), reward)
            score_history.append(score)
            env.destroy()
            if i > 30:
                avg_score = np.mean(score_history[-30:])

                if avg_score > best_score:
                    best_score = avg_score
                    agent.save_models()

            print('episode ', i, 'score %.1f' % score,
                    'average score %.1f' % avg_score,
                  "best score %.1f" % best_score,
                  observation_)

        # x = [i+1 for i in range(n_games)]
        # plot_learning_curve(x, score_history, filename)

        except UnboundLocalError:
            pass
        except (KeyboardInterrupt, RuntimeError, IndexError):
            env.destroy()
            print("""
            ERRORED
            """)
            x = False
            while x == False:
                answer = input("Would you like to save?(y/n) ")
                if answer.lower() == "y":
                    x = True
                    agent.save_models(i*FRAMES)
                elif answer.lower() == "n":
                    x = True
                else:
                    print("Please answer with y or n")
            quit()
        finally:
            env.world.apply_settings(original_settings)
            env.destroy()
