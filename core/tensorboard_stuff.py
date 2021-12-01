def run_tensorboard_server(tensorboardlogs_dir):
    import shlex, subprocess

    print(f"Checking tensorboards logs on {tensorboardlogs_dir}")
    args = shlex.split(f'tensorboard --logdir=./{tensorboardlogs_dir} --host localhost --port 8088')
    p = subprocess.Popen(args)

    print(f"Tensorflow listening on http://localhost:8088")
    return p


from stable_baselines3.common.callbacks import BaseCallback
class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        try:
            self.logger.record(key="train/reward", value=self.locals["rewards"][0])
        except BaseException:
            self.logger.record(key="train/reward", value=self.locals["reward"][0])
        return True