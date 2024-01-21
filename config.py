import configparser


def get_config():
    config = configparser.ConfigParser()
    config.read("config.ini")

    return {
        "batch_size": config.getint("train", "batch_size"),
        "num_epochs": config.getint("train", "num_epochs"),
        "lr": config.getfloat("train", "lr"),
        "seq_len": config.getint("train", "seq_len"),
        "d_model": config.getint("train", "d_model"),
        "lang_src": config.get("train", "lang_src"),
        "lang_tgt": config.get("train", "lang_tgt"),
        "model_folder": config.get("train", "model_folder"),
        "model_filename": config.get("train", "model_filename"),
        "preload": config.getboolean("train", "preload"),
        "tokenizer_file": config.get("train", "tokenizer_file"),
        "experiment_name": config.get("train", "experiment_name"),
    }
