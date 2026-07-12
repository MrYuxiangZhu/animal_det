import argparse
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import AnimalDetectionDataset, detection_collate
from src.models.clip_like import SimpleTokenizer
from src.models.grounding_dino_like import GroundingDINOAnimal
from src.models.grounding_loss import GroundingDetectionLoss
from src.trainers.common import select_device, set_seed
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.visualization import save_loss_curve


def run_epoch(model, text_tokens, loader, criterion, optimizer, device, train: bool) -> Dict[str, float]:
    model.train(train)
    sums = {"total": 0.0, "box": 0.0, "obj": 0.0, "cls": 0.0}
    steps = 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, targets, _ in tqdm(loader, desc="grounding_train" if train else "grounding_val"):
            images = images.to(device)
            box_raw, objectness, class_logits = model(images, text_tokens)
            loss, parts = criterion(box_raw, objectness, class_logits, targets)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
            for key in sums:
                sums[key] += parts[key]
            steps += 1
    return {key: value / max(steps, 1) for key, value in sums.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GroundingDINO-like animal open-vocabulary detector")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])
    logger = setup_logger("train_grounding_dino", cfg["project"]["log_dir"])
    device = select_device(cfg["train"]["device"])
    class_names = cfg["data"]["class_names"]
    grounding_cfg = cfg["grounding_dino"]

    train_set = AnimalDetectionDataset(cfg["data"]["root"], cfg["data"]["train_images"], cfg["data"]["train_labels"], cfg["data"]["image_size"])
    val_set = AnimalDetectionDataset(cfg["data"]["root"], cfg["data"]["val_images"], cfg["data"]["val_labels"], cfg["data"]["image_size"])
    train_loader = DataLoader(train_set, batch_size=grounding_cfg["batch_size"], shuffle=True, num_workers=cfg["data"]["num_workers"], collate_fn=detection_collate, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=grounding_cfg["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"], collate_fn=detection_collate, pin_memory=True)

    tokenizer = SimpleTokenizer(context_length=grounding_cfg["context_length"])
    prompts = [f"a photo of a {name}" for name in class_names]
    text_tokens = tokenizer.encode(prompts).to(device)
    model = GroundingDINOAnimal(tokenizer.vocab_size, grounding_cfg["context_length"], len(class_names), grounding_cfg["hidden_dim"], grounding_cfg["width_mult"]).to(device)
    criterion = GroundingDetectionLoss(len(class_names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=grounding_cfg["learning_rate"], weight_decay=cfg["train"]["weight_decay"])
    ckpt_dir = Path(cfg["project"]["output_dir"]) / "checkpoints" / "grounding_dino"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    history: Dict[str, List[float]] = {"train_total": [], "val_total": [], "train_box": [], "train_obj": [], "train_cls": []}

    for epoch in range(1, grounding_cfg["epochs"] + 1):
        train_loss = run_epoch(model, text_tokens, train_loader, criterion, optimizer, device, True)
        val_loss = run_epoch(model, text_tokens, val_loader, criterion, optimizer, device, False)
        for key in history:
            source = train_loss if key.startswith("train") else val_loss
            metric = key.split("_", 1)[1]
            history[key].append(source[metric])
        save_loss_curve(history, grounding_cfg["loss_curve"])
        logger.info("Epoch %03d | train total %.4f box %.4f obj %.4f cls %.4f | val total %.4f", epoch, train_loss["total"], train_loss["box"], train_loss["obj"], train_loss["cls"], val_loss["total"])
        ckpt = {"epoch": epoch, "model": model.state_dict(), "class_names": class_names, "tokenizer": {"context_length": tokenizer.context_length}, "config": cfg}
        if val_loss["total"] < best_val:
            best_val = val_loss["total"]
            torch.save(ckpt, ckpt_dir / "best.pt")
            logger.info("保存 GroundingDINO-like best checkpoint: %.4f", best_val)


if __name__ == "__main__":
    main()
