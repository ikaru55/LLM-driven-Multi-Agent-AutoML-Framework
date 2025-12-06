import os
import sys
import asyncio
import torch
import yaml
import json
import time
import subprocess
import shutil
import re
from typing import Tuple, Optional, Dict
from copy import deepcopy

from utils.validator import check_syntax, check_model_constraints
from utils.log_parser import parse_log
from utils.knowledge_base import load_papers
from utils.debate import (
    run_meeting,
    execute_and_debug_cycle,
    summarize_experiment,
    ensure_all_gems_available,
)
from utils.log_manager import CycleLogManager

CONFIG_PATH = "base_config.yaml"
HISTORY_PATH = "history.json"
LOG_DIR = "logs"
CORE_CODE_DIR = "core_code"
PAPER_DIR = "papers"

TARGET_FILES = ["model.py", "dataset_logic.py", "trainer_logic.py"]


def restore_skipped_stage_logs(start_stage: int, latest_dir: str, current_dir: str):
    """
    이전 사이클의 로그 중, 건너뛴(Skipped) 스테이지의 결과물을 현재 사이클로 복사합니다.
    """
    stage_dirs = {
        1: "stage_01_debate",
        2: "stage_02_config_patch",
        3: "stage_03_code_update",
        4: "experiment",
    }

    print(f"[Restoration] 📦 Copying logs for skipped stages (1 ~ {start_stage-1})...")

    for s in range(1, start_stage):
        folder_name = stage_dirs.get(s)
        if not folder_name:
            continue

        src_path = os.path.join(latest_dir, folder_name)
        dst_path = os.path.join(current_dir, folder_name)

        if os.path.exists(src_path):
            try:
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                print(f"  └─ 📋 Copied: {folder_name}")
            except Exception as e:
                print(f"  └─ ⚠️ Failed to copy {folder_name}: {e}")


def load_config():
    if not os.path.exists(CONFIG_PATH):
        return {
            "max_experiments": 1,
            "train": {},
            "constraints": {"max_params": 20000000},
        }
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_config_patch(config, patch):
    allowed_top = {"train", "constraints", "data"}
    for k, v in patch.items():
        if k in allowed_top and isinstance(v, dict):
            if k not in config:
                config[k] = {}
            config[k].update(v)
    return config


def try_extract_config_patch(summary_text):
    match = re.search(r"CONFIG_PATCH:\s*(\{.*\})", summary_text, re.DOTALL)
    if not match:
        return None
    blob = match.group(1).strip()
    try:
        return json.loads(blob)
    except Exception as e:
        print(f"[System] ⚠️ JSON Patch Decode Error: {e}")
        return None


def load_history():
    if os.path.exists(HISTORY_PATH):
        try:
            if os.path.getsize(HISTORY_PATH) == 0:
                return {"experiments": []}
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"[System] ⚠️ history.json is corrupted or empty. Resetting history.")
        except Exception:
            return {"experiments": []}
    return {"experiments": []}


def save_history(history):
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def extract_comprehensive_metrics(result: dict, cycle_logger=None) -> dict:
    metrics = {
        "eer": result.get("eer", 100.0),
        "val_acc": result.get("val_acc", 0.0),
        "params": result.get("params", 0),
        "flops": result.get("flops", 0),
        "train_time": f"{result.get('speed', 0):.1f}s" if "speed" in result else "N/A",
    }
    if metrics["params"] == 0:
        try:
            from core_code.model import get_model

            model = get_model()
            metrics["params"] = sum(p.numel() for p in model.parameters())
        except Exception:
            pass
    return metrics


def get_latest_log_dir(exclude_dir: Optional[str] = None) -> Optional[str]:
    if not os.path.exists(LOG_DIR):
        return None
    subdirs = [
        os.path.join(LOG_DIR, d)
        for d in os.listdir(LOG_DIR)
        if os.path.isdir(os.path.join(LOG_DIR, d))
    ]
    if exclude_dir:
        exclude_abs = os.path.abspath(exclude_dir)
        subdirs = [d for d in subdirs if os.path.abspath(d) != exclude_abs]

    if not subdirs:
        return None

    latest_dir = max(subdirs, key=os.path.getmtime)
    return latest_dir


def load_state_from_logs(
    target_stage: int, current_config: dict, current_cycle_dir: Optional[str] = None
) -> Tuple[Optional[Dict[str, str]], str, dict]:
    """
    Restore state from previous logs using stage outputs for code and config.
    """
    latest_dir = get_latest_log_dir(exclude_dir=current_cycle_dir)

    if not latest_dir:
        print("[System] ⚠️ No previous logs found to restore state. Starting fresh.")
        return None, "", current_config

    print(f"[System] 🔄 Restoring state from: {latest_dir}")

    restored_codes = {}
    restored_summary = ""
    restored_config = current_config

    if target_stage > 1:
        s1_path = os.path.join(latest_dir, "stage_01_debate", "output.json")
        if os.path.exists(s1_path):
            try:
                with open(s1_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    restored_codes = data.get("new_codes", {})
                    restored_summary = data.get("summary", "")
                print(
                    f"[State] ✅ Loaded {len(restored_codes)} files from Stage 1 output."
                )
            except Exception as e:
                print(f"[State] ⚠️ Failed to load Stage 1 output: {e}")

    if target_stage >= 4:
        s3_dir = os.path.join(latest_dir, "stage_03_code_update")
        if os.path.exists(s3_dir):
            print(f"[State] 📂 Found Stage 3 directory. Loading 'new_' files...")
            loaded_count = 0
            for filename in TARGET_FILES:
                new_filename = f"new_{filename}"
                file_path = os.path.join(s3_dir, new_filename)

                if os.path.exists(file_path):
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            restored_codes[filename] = f.read()
                            loaded_count += 1
                    except Exception as e:
                        print(f"[State] ⚠️ Failed to read {new_filename}: {e}")

            if loaded_count > 0:
                print(
                    f"[State] ✅ Overwrote {loaded_count} files using Stage 3 artifacts (new_*)."
                )

    if target_stage > 2:
        s2_path = os.path.join(latest_dir, "stage_02_config_patch", "output.json")
        if os.path.exists(s2_path):
            try:
                with open(s2_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    restored_config = data.get("patched_config", current_config)
                print("[State] ✅ Loaded config from Stage 2 output.")
            except Exception as e:
                print(f"[State] ⚠️ Failed to load Stage 2 output: {e}")

    return restored_codes, restored_summary, restored_config


def run_experiment(
    exp_id,
    config,
    cycle_logger=None,
    test_mode=False,
    fast_epochs=None,
    early_stop_patience=None,
    early_stop_delta=None,
):
    print(f"\n--- Starting Experiment {exp_id} ---")

    validation_results = {"syntax_check": {}, "constraint_check": None}

    print("[Executor] Validating code syntax for all files...")
    all_syntax_passed = True

    for filename in TARGET_FILES:
        file_path = os.path.join(CORE_CODE_DIR, filename)
        if not os.path.exists(file_path):
            print(f"⚠️ {filename} not found, skipping syntax check.")
            continue

        success, msg = check_syntax(file_path)
        validation_results["syntax_check"][filename] = {
            "success": success,
            "message": msg,
        }
        if not success:
            all_syntax_passed = False
            error_msg = f"Syntax Error in {filename}: {msg}"
            print(f"[Executor] ❌ {error_msg}")

    if not all_syntax_passed:
        if cycle_logger:
            cycle_logger.save_validation_log(
                {"validation_status": "failed_syntax", "details": validation_results}
            )
        return {"status": "Fail", "error": "Syntax Check Failed (see details)"}

    model_path = os.path.join(CORE_CODE_DIR, "model.py")
    if os.path.exists(model_path):
        success, msg = check_model_constraints(
            model_path, config["constraints"].get("max_params", 20000000)
        )
        validation_results["constraint_check"] = {"success": success, "message": msg}
        if not success:
            if cycle_logger:
                cycle_logger.save_validation_log(
                    {
                        "validation_status": "failed_constraints",
                        "details": validation_results,
                    }
                )
            return {"status": "Fail", "error": f"Constraint Violation: {msg}"}
        print(f"[Executor] Constraint Check Passed: {msg}")

    if cycle_logger:
        cycle_logger.save_validation_log(
            {"validation_status": "passed", "details": validation_results}
        )

    print("[Executor] Starting training...")
    log_file = os.path.join(LOG_DIR, f"exp_{exp_id}_train.log")

    train_payload = deepcopy(config)
    train_section = train_payload.setdefault("train", {})
    if fast_epochs is not None:
        train_section["epochs_full"] = fast_epochs
    if early_stop_patience is not None:
        train_section["early_stop_patience"] = early_stop_patience
    if early_stop_delta is not None:
        train_section["early_stop_delta"] = early_stop_delta

    train_config_str = json.dumps(train_payload)

    py_code = f"""
import sys
sys.path.insert(0, '.')
import json
from core_code.trainer import main as run_trainer

config = json.loads('''{train_config_str}''')
run_trainer(config)
"""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTHONIOENCODING", "utf-8")

    try:
        with open(log_file, "w", buffering=1, encoding="utf-8") as f:
            proc = subprocess.Popen(
                [sys.executable, "-u", "-c", py_code],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
            for line in proc.stdout:
                print(line, end="")
                f.write(line)
                f.flush()
            ret = proc.wait(timeout=3600)
            if ret != 0:
                tail = ""
                try:
                    with open(log_file, "r", encoding="utf-8", errors="replace") as rf:
                        tail = rf.read()[-4000:]
                except Exception:
                    pass
                return {
                    "status": "Fail",
                    "error": f"Training exited with code {ret}",
                    "trace_tail": tail,
                    "log_file": log_file,
                }
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except Exception:
            pass
        return {"status": "Fail", "error": "Training Timeout"}
    except Exception as e:
        return {"status": "Fail", "error": f"Execution Error: {str(e)}"}

    print("[Reporter] Parsing logs...")
    metrics = parse_log(log_file)
    if metrics.get("status") == "Fail":
        return metrics

    if "val_acc" in metrics:
        calculated_eer = 100.0 - metrics.get("val_acc", 0.0)
        if "eer" not in metrics:
            metrics["eer"] = calculated_eer

    try:
        with open(log_file, "r", encoding="utf-8") as f:
            log_content = f.read()

            best_epoch_match = re.search(r"\[Result\] Best Epoch: (\d+)", log_content)
            if best_epoch_match:
                metrics["best_epoch"] = int(best_epoch_match.group(1))

            best_eer_match = re.search(r"\[Result\] Best EER: ([\d\.]+)", log_content)
            if best_eer_match:
                real_eer = float(best_eer_match.group(1))
                metrics["eer"] = real_eer

            params_match = re.search(r"\[Metrics\] Params: (\d+)", log_content)
            if params_match:
                metrics["params"] = int(params_match.group(1))

            speed_matches = re.findall(
                r"\[Metrics\] Speed: ([\d\.]+)s/epoch", log_content
            )
            if speed_matches:
                speeds = [float(s) for s in speed_matches]
                metrics["speed"] = sum(speeds) / len(speeds)

            all_epochs = re.findall(r"Epoch (\d+)/(\d+)", log_content)
            if all_epochs:
                last_epoch_info = all_epochs[-1]
                metrics["stop_epoch"] = int(last_epoch_info[0])
                metrics["total_epochs"] = int(last_epoch_info[1])

    except Exception as e:
        print(f"[Reporter] ⚠️ Secondary parsing failed: {e}")

    return metrics


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="AutoML Main Executor")
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    parser.add_argument("--fast-epochs", type=int, help="Force override fast epochs")
    parser.add_argument(
        "--early-stop-patience", type=int, help="Force override early stop patience"
    )
    parser.add_argument(
        "--early-stop-delta", type=float, help="Force override early stop delta"
    )
    parser.add_argument(
        "--start-from", type=int, default=1, help="Start from specific stage."
    )
    parser.add_argument(
        "--forever",
        action="store_true",
        help="Run indefinitely ignoring max_experiments config.",
    )

    args = parser.parse_args()

    first_run = True

    print("\n[System] 🚀 AutoML Loop Started.")
    if args.forever:
        print("[System] ♾️ Running in FOREVER mode (Infinite Loop).")

    while True:
        config = load_config()
        history = load_history()

        if history["experiments"]:
            executed_count = max(
                int(exp.get("cycle_id", 0)) for exp in history["experiments"]
            )
        else:
            executed_count = 0

        target_max = int(config.get("max_experiments", 1))
        if not args.forever and executed_count >= target_max:
            if not (first_run and args.start_from > 1):
                print(
                    f"\n[System] 🛑 Reached max_experiments limit ({target_max}). Stopping."
                )
                print(
                    f"Tip: Increase 'max_experiments' in base_config.yaml or use --forever to continue."
                )
                break

        if first_run and args.start_from > 1:
            exp_id = executed_count + 1
            if exp_id == 0:
                exp_id = 1
            start_stage = args.start_from
            print(f"\n>>> [Cycle {exp_id}] 🔄 Resuming from Stage {start_stage}...")
        else:
            exp_id = executed_count + 1
            start_stage = 1
            print(f"\n>>> [Cycle {exp_id}] ✨ Initiating New Cycle...")

        if args.fast_epochs is not None:
            if "train" not in config:
                config["train"] = {}
            config["train"]["epochs_fast"] = args.fast_epochs
            if config.get("mode") == "fast" or args.test:
                config["train"]["epochs_full"] = args.fast_epochs

        if args.early_stop_patience is not None:
            if "train" not in config:
                config["train"] = {}
            config["train"]["early_stop_patience"] = args.early_stop_patience

        if args.early_stop_delta is not None:
            if "train" not in config:
                config["train"] = {}
            config["train"]["early_stop_delta"] = args.early_stop_delta

        if args.test:
            config["test_mode"] = True
            config["mode"] = "fast"
            if args.fast_epochs is None:
                config["train"]["epochs_fast"] = 1
                config["train"]["epochs_full"] = 1

        cycle_logger = CycleLogManager(exp_id, logs_root=LOG_DIR)
        current_log_dir = cycle_logger.get_cycle_dir()

        result = {"status": "Fail", "error": "Unknown error occurred during the cycle."}

        latest_dir = get_latest_log_dir(exclude_dir=current_log_dir)
        if start_stage > 1 and latest_dir:
            restore_skipped_stage_logs(start_stage, latest_dir, current_log_dir)
        new_codes = {}

        if start_stage > 1:
            s3_local_dir = os.path.join(current_log_dir, "stage_03_code_update")
            if os.path.exists(s3_local_dir):
                print(
                    f"[Restoration] 💾 Syncing code from copied logs to core_code/..."
                )
                for filename in TARGET_FILES:
                    code_path = os.path.join(s3_local_dir, f"new_{filename}")
                    if not os.path.exists(code_path):
                        code_path = os.path.join(s3_local_dir, f"original_{filename}")

                    if os.path.exists(code_path):
                        with open(code_path, "r", encoding="utf-8") as f:
                            code_content = f.read()
                            new_codes[filename] = code_content
                            target_path = os.path.join(CORE_CODE_DIR, filename)
                            with open(target_path, "w", encoding="utf-8") as tf:
                                tf.write(code_content)
                print(f"  └─ ✅ Synced files.")

            if start_stage > 4:
                metrics_local_path = os.path.join(
                    current_log_dir, "experiment", "metrics.json"
                )
                if os.path.exists(metrics_local_path):
                    try:
                        with open(metrics_local_path, "r") as f:
                            result = json.load(f)
                    except:
                        pass
        import gc

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        try:
            # ============================================================
            # Stage 01: Debate & Synthesis
            # ============================================================
            if start_stage <= 1:
                print("\n[Stage 01] Checking Agent Health...")
                if not await ensure_all_gems_available():
                    print("[System] 🛑 Critical Gem Error. Retrying in 60s...")
                    await asyncio.sleep(60)
                    if not await ensure_all_gems_available():
                        print("[System] ❌ FATAL: Gems unavailable. Exiting cycle.")
                        break
                print(f"\n[Stage 01] Debate & Synthesis")
                try:
                    paper_context = load_papers(PAPER_DIR)
                except:
                    paper_context = "No papers available."

                metrics_summary = "No prior experiments."
                current_history = load_history()
                if current_history["experiments"]:
                    last_exp = current_history["experiments"][-1]
                    metrics_summary = (
                        f"Last Metrics: EER={last_exp.get('eer', 100):.2f}%, "
                        f"Status={last_exp.get('status')}"
                    )

                new_codes = await run_meeting(
                    current_history,
                    metrics_summary,
                    paper_context,
                    config,
                    cycle_logger=cycle_logger,
                )

            else:
                print(f"\n[Stage 01] Skipped (Loaded from logs)")

            # ============================================================
            # Stage 02: Config Patch
            # ============================================================
            if start_stage <= 2:
                print(f"\n[Stage 02] Config Patch (Optimized via TrainerLogic)")
                cycle_logger.save_config_outputs(config, {}, config)
                print(
                    f"[Config] ⏩ Patching skipped. Hyperparameters are now managed by TrainerAdapter."
                )
            else:
                print(f"\n[Stage 02] Skipped")

            # ============================================================
            # Stage 03: Code Update
            # ============================================================
            if start_stage <= 3:
                print(f"\n[Stage 03] Code Update")
                for filename in TARGET_FILES:
                    fpath = os.path.join(CORE_CODE_DIR, filename)
                    if os.path.exists(fpath):
                        cycle_logger.save_original_model(fpath, filename)

                for filename in TARGET_FILES:
                    cycle_logger.save_new_model(new_codes, filename)

                if new_codes:
                    for filename, code_content in new_codes.items():
                        target_path = os.path.join(CORE_CODE_DIR, filename)
                        print(f"[System] 💾 Overwriting {filename}...")
                        with open(target_path, "w", encoding="utf-8") as f:
                            f.write(code_content)
                else:
                    print("[System] ⚠️ No new code generated.")
                    for filename in TARGET_FILES:
                        target_path = os.path.join(CORE_CODE_DIR, filename)
                        if not os.path.exists(target_path):
                            with open(target_path, "w") as f:
                                f.write("# Dummy Code")
            else:
                print(f"\n[Stage 03] Skipped")

            # ============================================================
            # Stage 04: Experiment Execution
            # ============================================================
            if start_stage <= 4:
                torch.cuda.empty_cache()
                result = await execute_and_debug_cycle(exp_id, config, cycle_logger)

                torch.cuda.empty_cache()
                gc.collect()

            # ============================================================
            # Stage 05: Final Summary & Model
            # ============================================================
            if start_stage <= 5:
                print(f"\n[Stage 05] Final Summary & Model")
                for filename in TARGET_FILES:
                    fpath = os.path.join(CORE_CODE_DIR, filename)
                    if os.path.exists(fpath):
                        cycle_logger.save_final_model(fpath)

                checkpoint_found = False
                for ext in [".pth", ".pkl", ".h5", ".onnx"]:
                    checkpoint_path = os.path.join(CORE_CODE_DIR, f"trained_model{ext}")

                    if not os.path.exists(checkpoint_path):
                        checkpoint_path = f"trained_model{ext}"

                    if os.path.exists(checkpoint_path):
                        cycle_logger.save_trained_model_checkpoint(checkpoint_path)
                        print(
                            f"[System] 💾 Saved trained model checkpoint: {checkpoint_path}"
                        )

                        cycle_logger.check_and_save_global_best(
                            history_path=HISTORY_PATH,
                            current_metrics=result,
                            source_checkpoint_path=checkpoint_path,
                        )

                        checkpoint_found = True
                        break

                if not checkpoint_found:
                    print(f"[System] ⚠️ Checkpoint NOT found.")
                cycle_summary = {
                    "cycle_id": exp_id,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "status": result.get("status", "unknown"),
                    "metrics": {
                        "eer": result.get("eer"),
                        "val_acc": result.get("val_acc"),
                        "train_acc": result.get("train_acc"),
                    },
                    "dynamics": {
                        "best_epoch": result.get("best_epoch", "N/A"),
                        "total_epochs": config.get("train", {}).get(
                            "epochs_full", "N/A"
                        ),
                        "final_grad_norm": result.get("grad_norm", "N/A"),
                        "final_train_loss": result.get("train_loss", "N/A"),
                        "final_val_loss": result.get("val_loss", "N/A"),
                    },
                    "efficiency": {
                        "params": result.get("params", "N/A"),
                        "speed": result.get("speed", "N/A"),
                    },
                }
                if "llm_summary_text" in locals() and llm_summary_text:
                    cycle_summary["hypothesis_summary"] = llm_summary_text

                cycle_logger.save_cycle_summary(cycle_summary)
                cycle_logger.finalize_cycle(result.get("status", "unknown"))

            else:
                print(f"\n[Stage 06] Skipped")

        except Exception as e:
            print(f"[System] ❌ Cycle {exp_id} fatal error: {e}")
            cycle_logger.finalize_cycle("error")
            import traceback

            traceback.print_exc()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

        # ============================================================
        # Final: History Update & Rollback Logic
        # ============================================================

        llm_summary_text = None
        if result.get("status") == "Success" or True:
            try:
                debate_log_path = os.path.join(
                    cycle_logger.dirs["debate"], "debate_log.json"
                )
                debate_data = {}
                if os.path.exists(debate_log_path):
                    with open(debate_log_path, "r", encoding="utf-8") as f:
                        debate_data = json.load(f)

                old_codes_content = {}
                new_codes_content = {}

                for fname in TARGET_FILES:
                    curr_p = os.path.join(CORE_CODE_DIR, fname)
                    if os.path.exists(curr_p):
                        with open(curr_p, "r", encoding="utf-8") as f:
                            new_codes_content[fname] = f.read()

                    old_p = os.path.join(cycle_logger.dirs["code"], f"original_{fname}")
                    if os.path.exists(old_p):
                        with open(old_p, "r", encoding="utf-8") as f:
                            old_codes_content[fname] = f.read()

                llm_summary_text = await summarize_experiment(
                    cycle_id=exp_id,
                    metrics=result,
                    old_codes=old_codes_content,
                    new_codes=new_codes_content,
                    debate_log=debate_data,
                )
                print(f"[Summary] 🧠 LLM Insight: {llm_summary_text}")
            except Exception as e:
                print(f"[Summary] ⚠️ Failed to generate LLM summary: {e}")

        cycle_logger.update_global_history(
            history_path=HISTORY_PATH,
            result=result,
            target_files=TARGET_FILES,
            core_code_dir=CORE_CODE_DIR,
            llm_summary=llm_summary_text,
        )

        if result.get("status") != "Success":
            print(
                f"[System] ⏪ Cycle {exp_id} Failed. Reverting code to pre-cycle state..."
            )
            revert_count = 0
            for filename in TARGET_FILES:
                backup_path = os.path.join(
                    cycle_logger.dirs["code"], f"original_{filename}"
                )
                target_path = os.path.join(CORE_CODE_DIR, filename)

                if os.path.exists(backup_path):
                    try:
                        shutil.copy(backup_path, target_path)
                        revert_count += 1
                    except Exception as e:
                        print(f"[System] ⚠️ Failed to revert {filename}: {e}")

            if revert_count > 0:
                print(f"[System] 🔄 Reverted {revert_count} files successfully.")

        metrics_val = result.get("eer", "N/A")
        print(
            f"[Result] Cycle {exp_id} Finished | Status: {result.get('status')} | EER: {metrics_val}%"
        )

        gc.collect()
        torch.cuda.empty_cache()

        first_run = False
        await asyncio.sleep(5)

    print("\n[System] All experiments finished.")


if __name__ == "__main__":
    asyncio.run(main())
