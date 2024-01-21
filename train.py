import torch
import torch.nn as nn

from dataset import BilingualDataset, causal_mask
from model import build_transformer
from config import get_config, get_weights_file_path

from datasets import load_dataset
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter


def get_all_sentences(ds, lang):
    for item in ds:
        yield item["translation"][lang]


def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if tokenizer_path.exists():
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    else:
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2
        )
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    return tokenizer


def get_ds(config):
    ds_raw = load_dataset(
        "opus_books", f'{config["lang_src"]}-{config["lang_tgt"]}', split="train"
    )

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["lang_tgt"])

    # Split into train and validation sets
    train_size = int(len(ds_raw) * config["train_size"])
    val_size = len(ds_raw) - train_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_size, val_size])

    # Build datasets
    train_ds = BilingualDataset(
        train_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )
    val_ds = BilingualDataset(
        val_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )

    max_len_src = 0
    max_len_tgt = 0

    for item in train_ds:
        src_ids = tokenizer_src.encode(item["src_text"]).ids
        tgt_ids = tokenizer_tgt.encode(item["tgt_text"]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max length of source text: {max_len_src}")
    print(f"Max length of target text: {max_len_tgt}")

    # Create dataloaders
    train_dl = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=True,
    )

    return train_dl, val_dl, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(
        vocab_src_len,
        vocab_tgt_len,
        config["seq_len"],
        config["seq_len"],
        config["d_model"],
    )
    return model


def train_model(config):
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ensure model folder exists
    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    # Get the dataloaders
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

    # Get the model
    model = get_model(
        config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()
    ).to(device)

    # Initialize tensorboard for monitoring
    writer = SummaryWriter(config["experiment_name"])

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    initial_epoch = 0
    global_step = 0

    # Load checkpoint if it exists
    if config["preload"]:
        model_filename = get_weights_file_path(config, config["preload"])
        print(f"Loading model weights from {model_filename}")
        checkpoint = torch.load(model_filename)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        initial_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["global_step"]
