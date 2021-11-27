def run_tensorboard_server(config):
    import shlex, subprocess

    args = shlex.split(f'tensorboard --logdir={config.TENSORBOARD_LOG_DIR} --host localhost --port 8088')
    p = subprocess.Popen(args)

    print(f"Tensorflow listening on http://localhost:8088")
    return p