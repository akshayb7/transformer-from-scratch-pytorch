import configparser
import Path


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
        "model_basename": config.get("train", "model_basename"),
        "preload": config.getboolean("train", "preload"),
        "tokenizer_file": config.get("train", "tokenizer_file"),
        "experiment_name": config.get("train", "experiment_name"),
    }


def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}_{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)
