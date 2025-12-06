import re
import os
import numpy as np


def parse_log(log_path):
    """Parse training log and extract metrics including EER, losses, and timing."""
    if not os.path.exists(log_path):
        return {"status": "Fail", "error": "Log file not found"}

    metrics = {
        "status": "Success",
        "eer": 100.0,
        "val_acc": 0.0,
        "train_loss": 99.9,
        "val_loss": 99.9,
        "train_acc": 0.0,
        "grad_norm": 0.0,
        "max_grad_norm": 0.0,
        "lr": 0.0,
        "speed": 0.0,
        "best_epoch": 0,
        "total_epochs": 0,
        "stop_epoch": 0,
        "train_loss_history": [],
        "val_loss_history": [],
        "error": None,
    }
    epoch_pattern = re.compile(r"Epoch\s+(\d+)/(\d+)\s+\|")
    patterns = {
        "train_loss": re.compile(r"Train Loss:\s+([\d\.]+)"),
        "train_acc": re.compile(r"Train Acc:\s+([\d\.]+)%"),
        "val_loss": re.compile(r"Val Loss:\s+([\d\.]+)"),
        "val_acc": re.compile(r"Val Acc:\s+([\d\.]+)%"),
        "grad_norm": re.compile(r"Grad:\s+([\d\.]+)"),
        "time": re.compile(r"Time:\s+([\d\.]+)s"),
        "val_eer": re.compile(r"Val EER:\s+([\d\.]+)%"),
    }

    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        val_history = []

        for line in lines:
            ep_match = epoch_pattern.search(line)
            if ep_match:
                current_epoch = int(ep_match.group(1))
                total_epochs = int(ep_match.group(2))
                metrics["stop_epoch"] = current_epoch
                metrics["total_epochs"] = total_epochs

                t_loss = patterns["train_loss"].search(line)
                t_acc = patterns["train_acc"].search(line)
                v_loss = patterns["val_loss"].search(line)
                grad = patterns["grad_norm"].search(line)
                time_val = patterns["time"].search(line)

                if t_loss:
                    val = float(t_loss.group(1))
                    metrics["train_loss"] = val
                    metrics["train_loss_history"].append(val)

                if t_acc:
                    metrics["train_acc"] = float(t_acc.group(1))

                if v_loss:
                    val = float(v_loss.group(1))
                    metrics["val_loss"] = val
                    metrics["val_loss_history"].append(val)
                    val_history.append(
                        {"epoch": current_epoch, "val_loss": val, "eer": 100.0}
                    )

                if grad:
                    g_val = float(grad.group(1))
                    metrics["grad_norm"] = g_val
                    metrics["max_grad_norm"] = max(metrics["max_grad_norm"], g_val)

                if time_val:
                    metrics["speed"] = float(time_val.group(1))

            eer_match = patterns["val_eer"].search(line)
            if eer_match:
                eer_val = float(eer_match.group(1))
                metrics["eer"] = eer_val
                if val_history:
                    val_history[-1]["eer"] = eer_val

            val_acc_match = patterns["val_acc"].search(line)
            if val_acc_match:
                metrics["val_acc"] = float(val_acc_match.group(1))

            if "Traceback" in line or "Error" in line or "Exception" in line:
                metrics["status"] = "Fail"
                metrics["error"] = "Runtime Error Detected"

        if val_history:
            has_eer = any(x["eer"] < 100.0 for x in val_history)

            if has_eer:
                best_record = min(val_history, key=lambda x: x["eer"])
            else:
                best_record = min(val_history, key=lambda x: x["val_loss"])

            metrics["best_epoch"] = best_record["epoch"]
            if metrics["eer"] == 100.0 and best_record["eer"] < 100.0:
                metrics["eer"] = best_record["eer"]

        return metrics

    except Exception as e:
        return {"status": "Fail", "error": f"Parsing Error: {str(e)}"}
