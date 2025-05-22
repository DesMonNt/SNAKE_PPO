import random
from snake_agent import Wrapper
from snake_game import SnakeEnv
from snake_trainer.trainer import Trainer


def make_env():
    grid_size = random.choice([n for n in range(6, 26, 2)])
    num_food = random.randint(1, 20)

    return Wrapper(SnakeEnv(grid_size=grid_size, num_food=num_food))


if __name__ == "__main__":
    trainer = Trainer(
        env_fn=make_env,
        num_envs=4,
        batch_size=256,
        num_episodes=2000
    ).train()
