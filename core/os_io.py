def create_folders(datasets_dir, trained_models_dir, tensorboardlogs_dir, results_dir):
    import os

    if not os.path.exists(datasets_dir):
        os.makedirs(datasets_dir)
    if not os.path.exists(trained_models_dir):
        os.makedirs(trained_models_dir)
    if not os.path.exists(tensorboardlogs_dir):
        os.makedirs(tensorboardlogs_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)