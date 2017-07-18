import curses
import fruit_env
import sys

def human_main(cwin):
    env = fruit_env.FruitCollectEnv(visualize=True)

    repeat = True
    while repeat:
        observation = env.reset()
        done = False
        while not done:
            env.render()
            action = -1
            input_key = sys.stdin.read(1)
            if input_key == 'w': action = 3
            if input_key == 'a': action = 1
            if input_key == 's': action = 0
            if input_key == 'd': action = 2
            if input_key == 'q':
                repeat = False
                done = True
            if action != -1:
                observation, reward, done, info = env.step(action)

curses.wrapper(human_main)
