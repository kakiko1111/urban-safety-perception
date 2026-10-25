"""Train and evaluate the RGB/grayscale x ResNet50/ViT experiments on Place Pulse 2.0.

Example:
    python run_experiments.py --data-path data/place-pulse-2.0
    python run_experiments.py --data-path data/place-pulse-2.0 --experiments vit_rgb --epochs 5
"""

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
import torch

from safety_perception.config import DEFAULTS, DISPLAY_NAMES, EXPERIMENTS, pick_device
from safety_perception.data import build_dataset, build_filename_map, make_loader, place_split
from safety_perception.evaluate import get_predictions, split_errors
from safety_perception.models import build_model, count_parameters
from safety_perception.plots import plot_confusion_matrices, plot_error_examples, plot_training_summary
from safety_perception.train import train_model


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-path", type=Path, default=Path("data/place-pulse-2.0"),
                   help="Folder with studies.tsv, qscores.tsv, locations.tsv and images/")
    p.add_argument("--output", type=Path, default=Path("outputs"))
    p.add_argument("--experiments", nargs="+", choices=list(EXPERIMENTS), default=list(EXPERIMENTS))
    p.add_argument("--epochs", type=int, default=DEFAULTS["num_epochs"])
    p.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--device", default=None, help="mps, cuda or cpu (auto-detected if omitted)")
    p.add_argument("--no-pretrained", action="store_true",
                   help="Random init instead of ImageNet weights (for quick tests only)")
    p.add_argument("--skip-error-examples", action="store_true",
                   help="Don't save figures of misclassified street-view images")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = {**DEFAULTS, "num_epochs": args.epochs, "batch_size": args.batch_size}
    device = args.device or pick_device()
    torch.manual_seed(cfg["seed"])

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    images_path = args.data_path / "images"
    print(f"Device: {device}")

    # 1. Data
    id_to_filename = build_filename_map(images_path)
    df = build_dataset(args.data_path, id_to_filename, cfg["min_votes"], cfg["threshold_std"])
    df.to_csv(out / "training_dataset.csv", index=False)
    with open(out / "image_id_to_filename.pkl", "wb") as f:
        pickle.dump(id_to_filename, f)

    # 2. Split by place (no place in both train and val)
    train_df, val_df = place_split(df, cfg["val_fraction"], cfg["seed"])
    split_info = {
        "train_places": sorted(map(str, train_df["place_id"].unique())),
        "val_places": sorted(map(str, val_df["place_id"].unique())),
        "train_images": len(train_df),
        "val_images": len(val_df),
    }
    (out / "split.json").write_text(json.dumps(split_info, indent=2))

    # 3. Train each experiment
    histories, best_accs, analysis = {}, {}, {}
    for name in args.experiments:
        arch, color_mode = EXPERIMENTS[name]
        print(f"\n>>> {DISPLAY_NAMES[name]}")
        train_loader = make_loader(train_df, images_path, id_to_filename, color_mode, cfg, train=True)
        val_loader = make_loader(val_df, images_path, id_to_filename, color_mode, cfg, train=False)

        model = build_model(arch, pretrained=not args.no_pretrained).to(device)
        print(f"Parameters: {count_parameters(model):,}")
        model, history, best_acc = train_model(model, train_loader, val_loader, name, cfg, device, out)
        histories[name], best_accs[name] = history, best_acc
        print(f"Best validation accuracy: {best_acc:.2f}%")

        id_loader = make_loader(val_df, images_path, id_to_filename, color_mode, cfg, train=False, return_id=True)
        preds, labels, image_ids, probs = get_predictions(model, id_loader, device)
        analysis[name] = {"preds": preds, "labels": labels, "image_ids": image_ids, "probs": probs}

    # 4. Results tables
    results = pd.DataFrame({
        "experiment": list(best_accs),
        "preprocessing": [EXPERIMENTS[k][1] for k in best_accs],
        "model": [EXPERIMENTS[k][0] for k in best_accs],
        "val_accuracy": list(best_accs.values()),
    })
    results.to_csv(out / "all_results.csv", index=False)
    print("\n" + results.to_string(index=False))

    history_rows = []
    for name, hist in histories.items():
        for epoch in range(len(hist["train_acc"])):
            history_rows.append({"experiment": name, "epoch": epoch + 1,
                                 **{k: v[epoch] for k, v in hist.items()}})
    pd.DataFrame(history_rows).to_csv(out / "training_history.csv", index=False)

    if set(EXPERIMENTS) <= set(best_accs):
        a = best_accs
        rgb, gray = (a["resnet50_rgb"] + a["vit_rgb"]) / 2, (a["resnet50_gray"] + a["vit_gray"]) / 2
        resnet, vit = (a["resnet50_rgb"] + a["resnet50_gray"]) / 2, (a["vit_rgb"] + a["vit_gray"]) / 2
        print(f"\nRGB vs grayscale: {rgb:.2f}% vs {gray:.2f}% (color contribution {rgb - gray:+.2f})")
        print(f"ResNet50 vs ViT: {resnet:.2f}% vs {vit:.2f}% (ViT minus ResNet {vit - resnet:+.2f})")

    # 5. Figures
    plot_training_summary(histories, best_accs, out)
    plot_confusion_matrices(analysis, out)

    if not args.skip_error_examples:
        best = max(best_accs, key=best_accs.get)
        r = analysis[best]
        errors, fps, fns = split_errors(r["preds"], r["labels"])
        print(f"\n{DISPLAY_NAMES[best]} errors: {len(errors)} / {len(r['labels'])} "
              f"(false positives {len(fps)}, false negatives {len(fns)})")
        common = (r["labels"], r["preds"], r["probs"], r["image_ids"], id_to_filename, images_path)
        plot_error_examples(fps, *common, "False positives: predicted SAFE, actually UNSAFE",
                            out / "error_false_positives.png")
        plot_error_examples(fns, *common, "False negatives: predicted UNSAFE, actually SAFE",
                            out / "error_false_negatives.png")

    print(f"\nDone. Outputs in {out}/")


if __name__ == "__main__":
    main()
