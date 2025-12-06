import os
import json
import shutil
from datetime import datetime
from typing import Dict, Any, Optional, List
import difflib


class CycleLogManager:
    """Manage per-cycle log directories and persistence for artifacts."""

    def __init__(self, cycle_id: int, logs_root: str = "logs"):
        self.cycle_id = cycle_id
        self.logs_root = logs_root
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.cycle_dir = os.path.join(
            logs_root, f"cycle_{cycle_id:03d}_{self.timestamp}"
        )

        self.dirs = {
            "root": self.cycle_dir,
            "debate": os.path.join(self.cycle_dir, "stage_01_debate"),
            "config": os.path.join(self.cycle_dir, "stage_02_config_patch"),
            "code": os.path.join(self.cycle_dir, "stage_03_code_update"),
            "experiment": os.path.join(self.cycle_dir, "stage_04_experiment"),
            "debug": os.path.join(self.cycle_dir, "stage_05_debug"),
            "final": os.path.join(self.cycle_dir, "stage_06_final"),
            "models": os.path.join(self.cycle_dir, "trained_models"),
        }

        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)

        self._init_cycle_info()

    def _init_cycle_info(self):
        cycle_info = {
            "cycle_id": self.cycle_id,
            "start_time": datetime.now().isoformat(),
            "status": "in_progress",
        }
        self._save_json(os.path.join(self.cycle_dir, "cycle_info.json"), cycle_info)

    @staticmethod
    def _save_json(filepath: str, data: Dict[str, Any], ensure_ascii: bool = False):
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=ensure_ascii)

    @staticmethod
    def _copy_file(src: str, dst: str):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)

    def save_debate_outputs(self, response: str, new_codes: Dict[str, str]):
        filepath = os.path.join(self.dirs["debate"], "output.json")
        data = {
            "timestamp": datetime.now().isoformat(),
            "response": response,
            "new_codes": new_codes,
        }
        self._save_json(filepath, data)
        return filepath

    def save_debate_log(self, debate_log: Dict[str, Any]):
        self._save_json(
            os.path.join(self.dirs["debate"], "debate_log.json"), debate_log
        )

    def save_config_outputs(self, original: dict, patch: dict, patched: dict):
        data = {
            "timestamp": datetime.now().isoformat(),
            "original": original,
            "patch": patch,
            "patched": patched,
        }
        self._save_json(os.path.join(self.dirs["config"], "output.json"), data)

    def save_original_model(self, model_path: str, filename: str):
        if os.path.exists(model_path):
            self._copy_file(
                model_path, os.path.join(self.dirs["code"], f"original_{filename}")
            )

    def save_new_model(self, new_codes: dict, filename: str):
        if filename in new_codes:
            with open(
                os.path.join(self.dirs["code"], f"new_{filename}"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(new_codes[filename])

    def save_validation_log(self, res: Dict[str, Any]):
        self._save_json(
            os.path.join(self.dirs["experiment"], "validation_log.json"), res
        )

    def save_debug_attempt(self, num: int, data: Dict[str, Any]):
        self._save_json(
            os.path.join(self.dirs["debug"], f"debug_attempt_{num}.json"), data
        )

    def save_final_model(self, model_path: str):
        if os.path.exists(model_path):
            self._copy_file(
                model_path,
                os.path.join(self.dirs["final"], os.path.basename(model_path)),
            )

    def save_trained_model_checkpoint(self, checkpoint_path: str):
        if os.path.exists(checkpoint_path):
            self._copy_file(
                checkpoint_path,
                os.path.join(self.dirs["models"], os.path.basename(checkpoint_path)),
            )

    def save_cycle_summary(self, summary_data: Dict[str, Any]):
        self._save_json(
            os.path.join(self.dirs["final"], "cycle_summary.json"), summary_data
        )

    def finalize_cycle(self, final_status: str):
        path = os.path.join(self.cycle_dir, "cycle_info.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                info = json.load(f)
            info.update(
                {"status": final_status, "end_time": datetime.now().isoformat()}
            )
            self._save_json(path, info)
        except:
            pass

    def get_cycle_dir(self) -> str:
        return self.cycle_dir

    def update_global_history(
        self,
        history_path: str,
        result: Dict[str, Any],
        target_files: List[str],
        core_code_dir: str,
        llm_summary: Optional[str] = None,
    ):
        """Append current cycle metrics to history.json."""

        history = {"experiments": []}
        if os.path.exists(history_path):
            try:
                with open(history_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    if content.strip():
                        history = json.loads(content)
            except Exception as e:
                print(f"[LogManager] ⚠️ Failed to load history: {e}")

        hypothesis = "Automated Optimization"
        if llm_summary:
            hypothesis = llm_summary
        else:
            debate_log_path = os.path.join(self.dirs["debate"], "debate_log.json")
            if os.path.exists(debate_log_path):
                try:
                    with open(debate_log_path, "r", encoding="utf-8") as f:
                        dlog = json.load(f)
                        if "Hacker" in dlog.get("strategy", {}):
                            hypothesis = dlog["strategy"]["Hacker"][:200]
                except:
                    pass

        exp_record = {
            "cycle_id": self.cycle_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": result.get("status", "Fail"),
            "hypothesis_summary": hypothesis,
            "metrics": {
                "eer": result.get("eer", 100.0),
                "val_acc": result.get("val_acc", 0.0),
                "train_acc": result.get("train_acc", 0.0),
            },
            "dynamics": {
                "best_epoch": result.get("best_epoch", "N/A"),
                "stop_epoch": result.get("stop_epoch", "N/A"),
                "total_epochs": result.get("total_epochs", "N/A"),
                "final_grad_norm": result.get("grad_norm", 0.0),
                "max_grad_norm": result.get("max_grad_norm", 0.0),
                "final_train_loss": result.get("train_loss", 0.0),
                "final_val_loss": result.get("val_loss", 0.0),
            },
            "efficiency": {
                "params": result.get("params", "N/A"),
                "speed": f"{result.get('speed', 0.0):.2f}s/epoch",
            },
        }
        history["experiments"].append(exp_record)
        self._save_json(history_path, history)
        print(
            f"[LogManager] 📘 Global history updated (Cycle {self.cycle_id}) with Dynamics & Efficiency."
        )

    def check_and_save_global_best(
        self,
        history_path: str,
        current_metrics: Dict[str, Any],
        source_checkpoint_path: str,
    ):
        """Copy checkpoint to final folder when current EER beats historical best."""
        current_eer = current_metrics.get("eer", 100.0)
        if current_eer >= 100.0:
            return False
        best_eer_so_far = 100.0
        if os.path.exists(history_path):
            try:
                with open(history_path, "r", encoding="utf-8") as f:
                    history = json.load(f)
                    experiments = history.get("experiments", [])
                    valid_eers = [
                        e.get("metrics", {}).get("eer", 100.0)
                        for e in experiments
                        if e.get("cycle_id") != self.cycle_id
                    ]
                    if valid_eers:
                        best_eer_so_far = min(valid_eers)
            except Exception as e:
                print(f"[LogManager] ⚠️ History read failed: {e}")

        if current_eer < best_eer_so_far:
            print(
                f"\n[LogManager] 🏆 NEW GLOBAL BEST! (Current: {current_eer:.2f}% < Previous Best: {best_eer_so_far:.2f}%)"
            )
            print(f"[LogManager] 💾 Saving Best Checkpoint to stage_06_final...")

            if os.path.exists(source_checkpoint_path):
                filename = f"BEST_cycle_{self.cycle_id:03d}_eer_{current_eer:.2f}.pth"
                dst_path = os.path.join(self.dirs["final"], filename)
                self._copy_file(source_checkpoint_path, dst_path)
                best_info = {
                    "is_global_best": True,
                    "cycle_id": self.cycle_id,
                    "eer": current_eer,
                    "beat_previous_best": best_eer_so_far,
                    "timestamp": self.timestamp,
                }
                self._save_json(
                    os.path.join(self.dirs["final"], "best_model_info.json"), best_info
                )
                return True
            else:
                print(
                    "[LogManager] ⚠️ Checkpoint file not found, cannot save best model."
                )

        return False
