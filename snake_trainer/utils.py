def worker(remote, env_fn):
    env = env_fn()

    while True:
        cmd, data = remote.recv()

        if cmd == "step":
            action = data
            next_obs, reward, done = env.step(action)

            if done:
                env = env_fn()
                next_obs = env.reset()

            remote.send((next_obs, reward, done))

        elif cmd == "reset":
            env = env_fn()
            obs = env.reset()
            remote.send(obs)

        elif cmd == "close":
            break
