import os
import sys
import re
import time
import ast
import asyncio
import shutil
import textwrap
import io
import traceback
from typing import Tuple, Optional, Dict, Any
from contextlib import redirect_stdout, redirect_stderr

from utils.gemini_web_api_helper import (
    generate_gemini_response,
    log_llm_interaction,
    fetch_gems,
    get_gem_by_id,
    get_gem_by_name,
    generate_with_gem,
    chat_with_gem_response,
)
from utils.log_parser import parse_log
from core_code import trainer

CORE_CODE_DIR = "core_code"
LOG_DIR = "logs"
MODEL_FILE_PATH = os.path.join(CORE_CODE_DIR, "model.py")
DATASET_FILE_PATH = os.path.join(CORE_CODE_DIR, "dataset_logic.py")
TRAINER_FILE_PATH = os.path.join(CORE_CODE_DIR, "trainer_logic.py")
TARGET_FILES = ["model.py", "dataset_logic.py", "trainer_logic.py"]
REQUEST_INTERVAL = 120

ROLE_GEM_TARGETS = {
    "Theorist": {
        "id": "1c6adef772d1",
        "names": ["The Theorist"],
    },
    "Hacker": {
        "id": "038d1d275f11",
        "names": ["The Hacker"],
    },
    "DatasetArchitect": {
        "id": "a7744404adde",
        "names": ["Dataset Architect"],
    },
    "ModelArchitect": {
        "id": "1d3e0cdd9a0c",
        "names": ["The Synthesizer"],
    },
    "TrainerArchitect": {
        "id": "9f6704833f5c",
        "names": ["Trainer Architect"],
    },
    "AutoDebugger": {
        "id": "f83ff3689e53",
        "names": ["Auto Debugger"],
    },
}
_role_gem_cache: dict = {}


def extract_code_block(response_text: str) -> str:
    """Extract a single code block from a response text."""
    import re

    if not response_text:
        return ""

    match = re.search(
        r"```python\s*(.*?)\s*```", response_text, re.DOTALL | re.IGNORECASE
    )
    if match:
        return match.group(1).strip()

    match = re.search(r"```\s*(.*?)\s*```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return response_text.strip()


def clean_code_comments(source_code: str) -> str:
    """Strip comments and docstrings from Python code and return normalized code."""
    try:
        parsed = ast.parse(source_code)
        return ast.unparse(parsed)
    except Exception as e:
        print(f"[Warning] Code cleaning failed (Syntax Error?): {e}")
        return source_code


def read_file_content(file_path: str, clean: bool = False) -> str:
    """Read file contents; return empty string if missing."""
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            if clean:
                return clean_code_comments(content)
            return content
    return f"# {file_path} not found."


def extract_code_from_response(response_text: str, label: str = "Code") -> str:
    """Extract Python code from an LLM response."""
    if not response_text:
        return None

    code_match = re.search(
        r"```python\s*(.*?)\s*```", response_text, re.DOTALL | re.IGNORECASE
    )
    if code_match:
        print(f"[{label}] ✅ Code extracted from ```python ... ``` block")
        return textwrap.dedent(code_match.group(1)).strip()

    code_match_generic = re.search(r"```\s*(.*?)\s*```", response_text, re.DOTALL)
    if code_match_generic:
        print(f"[{label}] ✅ Code extracted from generic ``` ... ``` block")
        return textwrap.dedent(code_match_generic.group(1)).strip()

    print(f"[{label}] ⚠️ Failed to extract code pattern.")
    return None


def _format_prompt(
    system_instruction: str, user_prompt: str, include_instruction: bool = True
) -> str:
    if include_instruction and system_instruction:
        return (
            f"System Instruction:\n{system_instruction}\n\nUser Query:\n{user_prompt}"
        )
    return user_prompt


async def _get_role_gem(role: str):
    """Load the Gem object mapped to the requested role."""
    target = ROLE_GEM_TARGETS.get(role)
    if not target:
        print(f"[System] ⚠️ No configuration found for role: {role}")
        return None

    if role in _role_gem_cache:
        return _role_gem_cache[role]

    gem = None
    max_retries = 3
    retry_delay = 20

    for attempt in range(1, max_retries + 1):
        try:
            print(
                f"[Gemini] 🔍 Searching for Gem (Attempt {attempt}/{max_retries}): Role={role}"
            )
            await fetch_gems(include_hidden=True)
            for name in target.get("names", []):
                if not name:
                    continue
                gem = await get_gem_by_name(name)
                if gem:
                    break
            if not gem and target.get("id"):
                gem = await get_gem_by_id(target["id"])

            if gem:
                _role_gem_cache[role] = gem
                gem_name = getattr(gem, "display_name", None) or getattr(
                    gem, "name", None
                )
                print(f"[Gemini] ✅ Loaded gem '{gem_name}' for role {role}")
                return gem

            if attempt < max_retries:
                await asyncio.sleep(retry_delay)

        except Exception as exc:
            print(f"[Gemini] ⚠️ Error lookup ({attempt}): {exc}")
            await asyncio.sleep(retry_delay)

    print(f"[Gemini] ❌ Failed to load Gem for role {role}")
    return None


async def ensure_all_gems_available() -> bool:
    """Verify all required Gems are available, retrying after cache reset if missing."""
    print("\n[System] 🛡️ Verifying availability of all required Gems...")

    missing_roles = []
    _role_gem_cache.clear()

    try:
        await fetch_gems(include_hidden=True)
    except Exception as e:
        print(f"[System] ⚠️ Initial fetch failed: {e}")

    for role in ROLE_GEM_TARGETS.keys():
        gem = await _get_role_gem(role)
        if not gem:
            missing_roles.append(role)

    if missing_roles:
        print(
            f"[System] ⚠️ Missing Gems for: {missing_roles}. Attempting Deep Refresh..."
        )
        await asyncio.sleep(5)
        _role_gem_cache.clear()
        try:
            await fetch_gems(include_hidden=True)
        except Exception as e:
            print(f"[System] ❌ Refresh failed: {e}")
            return False

        still_missing = []
        for role in missing_roles:
            gem = await _get_role_gem(role)
            if not gem:
                still_missing.append(role)

        if still_missing:
            print(
                f"[System] ❌ CRITICAL: The following Gems are still missing: {still_missing}"
            )
            print(
                f"[System] 💡 Please check your Gem IDs in ROLE_GEM_TARGETS or Internet Connection."
            )
            return False

    print("[System] ✅ All Gems verified and ready.")
    return True


async def _single_turn_request(
    system_instruction: str, user_prompt: str, model_name: str, gem=None
) -> str:
    """Send a single-turn request with cooldown handling."""
    response_text = None
    print(f"[System] Before Request Cooldown: Waiting {REQUEST_INTERVAL-60}s...")
    await asyncio.sleep(REQUEST_INTERVAL - 60)
    try:
        if gem:
            final_prompt = _format_prompt(
                system_instruction, user_prompt, include_instruction=False
            )
            response_text = await generate_with_gem(
                final_prompt, gem=gem, model=model_name
            )
        else:
            response_text = await generate_gemini_response(
                user_prompt=user_prompt,
                system_instruction=system_instruction,
                model=model_name,
            )
    except Exception as e:
        print(f"[API Error] {e}")
        return None

    print(f"[System] Response received. Cooldown: Waiting {REQUEST_INTERVAL}s...")
    await asyncio.sleep(REQUEST_INTERVAL)
    return response_text


async def get_agent_response(
    role: str,
    system_instruction: str,
    user_prompt: str,
    model_name: str = "gemini-3.0-pro",
    max_retries: int = 3,
) -> str:
    """Get an agent response with retry logic."""
    print(f"\n[{role}] Thinking... (Model: {model_name})")

    gem = await _get_role_gem(role)
    if not gem:
        print(f"[{role}] ⚠️ Gem not found, using direct API call.")
        exit(1)

    response_text = ""

    for attempt in range(1, max_retries + 1):
        try:
            print(f"⏳ [{role}] Attempt {attempt}/{max_retries}...")
            response_text = await _single_turn_request(
                system_instruction, user_prompt, model_name, gem
            )
            if response_text:
                print(f"✅ [{role}] Response received ({len(response_text)} chars)")
                break
            else:
                print(f"⚠️ [{role}] Empty response, retrying...")
                raise ValueError("Empty response received from API")
        except Exception as e:
            print(f"⚠️ [{role}] Attempt {attempt}/{max_retries} failed: {e}")

            if attempt < max_retries:
                wait_time = 2 ** (attempt - 1)
                print(f"⏳ [{role}] Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                print(f"\n❌ [{role}] API Failed after {max_retries} attempts")
                response_text = ""

    log_llm_interaction(
        role,
        system_instruction,
        user_prompt,
        response_text,
        bool(response_text),
        model_name,
    )
    return response_text


def parse_multi_file_response(response_text: str) -> Dict[str, str]:
    """
    Parses the response to extract multiple files.
    Expected format in response:
    <<<<FILE: model.py>>>>
    ... content ...
    <<<<END: model.py>>>>
    """
    extracted_files = {}

    pattern = re.compile(
        r"<<<<FILE:\s*([a-zA-Z0-9_]+\.py)>>>>(.*?)<<<<END:\s*\1>>>>", re.DOTALL
    )

    matches = pattern.findall(response_text)
    for filename, content in matches:
        clean_content = content.strip()
        clean_content = re.sub(r"^```python\s*", "", clean_content)
        clean_content = re.sub(r"^```\s*", "", clean_content)
        clean_content = re.sub(r"```$", "", clean_content)

        extracted_files[filename] = clean_content.strip()
        print(f"[Parser] 📦 Extracted {filename} ({len(clean_content)} chars)")

    return extracted_files


def format_history_for_context(history: dict, max_recent: int = 5) -> str:
    """실험 기록 포맷팅 (기존 함수 유지)"""
    if not history or "experiments" not in history or len(history["experiments"]) == 0:
        return "No previous experiments."

    experiments = history["experiments"][-max_recent:]
    context = (
        "\n[📊 EXPERIMENT HISTORY - Recent Performance Trends]\n" + "=" * 60 + "\n"
    )

    for exp in experiments:
        metrics = exp.get("metrics", {}) or {"eer": exp.get("eer", "N/A")}
        context += (
            f"Cycle {exp.get('cycle_id', 'N/A')}: EER: {metrics.get('eer', 'N/A')}%\n"
        )
        if exp.get("hypothesis"):
            context += f"  • Hypothesis: {exp['hypothesis']}\n"
        context += "-" * 60 + "\n"
    return context


async def run_meeting(history, metrics_summary, papers, config, cycle_logger=None):
    """
    Refactored Multi-Agent Debate Pipeline:
    1. Theorist & Hacker: Brainstorm Strategy
    2. DatasetArchitect: Update dataset_logic.py based on Strategy
    3. ModelArchitect: Update model.py based on Strategy + New Dataset Code
    4. TrainerArchitect: Update trainer_logic.py based on Strategy + New Dataset/Model Code
    """
    print("\n=== 🤖 Sequential Multi-Agent Pipeline Started ===")

    model_name = config.get("llm", {}).get("model", "gemini-3.0-pro")
    max_params = 11500000

    current_dataset_code = read_file_content(DATASET_FILE_PATH, clean=True)
    current_model_code = read_file_content(MODEL_FILE_PATH, clean=True)
    current_trainer_code = read_file_content(TRAINER_FILE_PATH, clean=True)

    history_context = format_history_for_context(history, max_recent=5)

    base_context = (
        f"🎯 [GOAL] Minimize EER for Face Verification (2882 classes).\n"
        f"⚙️  [CONSTRAINT] Max Parameters: {max_params/1e6:.1f}M\n"
        f"📊 [CURRENT PERFORMANCE]\n{metrics_summary}\n\n"
    )

    theorist_prompt = (
        f"### [CURRENT EXPERIMENT STATUS]\n"
        f"**Metrics**: {metrics_summary}\n"
        f"**History**: {history_context}\n\n"
        f"### [CURRENT CODEBASE]\n"
        f"1. **Dataset Logic**:\n```python\n{current_dataset_code}\n```\n"
        f"2. **Model Architecture**:\n```python\n{current_model_code}\n```\n"
        f"3. **Training Logic**:\n```python\n{current_trainer_code}\n```\n\n"
        f"**[TASK]**\n"
        f"Analyze the code and metrics above based on your system instructions."
        f"Identify bottlenecks in Data, Model, and Training interactions."
        f"Propose your **Master Plan** to minimize EER."
    )
    theorist_msg = await get_agent_response("Theorist", "", theorist_prompt, model_name)

    hacker_prompt = (
        f"{base_context}\n"
        f"**History**: {history_context}\n\n"
        f"[THEORIST'S MASTER PLAN]\n{theorist_msg}\n\n"
        f"### [TASK: HACKER]\n"
        f"You are the **Lead Engineer (Hacker)**. Critique the Theorist's plan for **technical feasibility and implementation details**.\n"
        f"1. **Verify Constraints**: Will the proposed model stay under 11.5M params?\n"
        f"2. **Refine Logic**: If Theorist suggests 'better augmentation', specify exact transforms (e.g., Mixup, RandAugment).\n"
        f"3. **Training Stability**: Warn about potential NaN issues or gradient explosions with the proposed Loss/Optimizer.\n\n"
        f"**Output a refined, concrete execution plan for the Architects.**"
    )
    hacker_msg = await get_agent_response("Hacker", "", hacker_prompt, model_name)

    print(f"✅ [Phase 1] Strategy Established. Cooldown 60s...")
    await asyncio.sleep(60)
    strategic_plan = (
        f"✅ [AGREED STRATEGY]\n"
        f"--- THEORIST ---\n{theorist_msg}\n"
        f"--- HACKER ---\n{hacker_msg}\n"
    )

    print("\n[🌊 Pipeline] Step 1: Modifying Dataset Logic...")
    dataset_prompt = (
        f"{base_context}\n"
        f"{strategic_plan}\n\n"
        f"[CURRENT dataset_logic.py]\n```python\n{current_dataset_code}\n```\n\n"
        f"[TASK] Rewrite `dataset_logic.py` to implement the data-related parts of the strategy.\n"
        f"Must be compatible with dataset.py which will handle this code in framework."
    )
    dataset_response = await get_agent_response(
        "DatasetArchitect", "", dataset_prompt, model_name
    )
    new_dataset_code = (
        extract_code_from_response(dataset_response, "DatasetArchitect")
        or current_dataset_code
    )

    print(f"✅ [Phase 2-1] Dataset Logic Updated. Cooldown 60s...")
    await asyncio.sleep(60)

    print("\n[🌊 Pipeline] Step 2: Modifying Model Architecture...")
    model_prompt = (
        f"{base_context}\n"
        f"{strategic_plan}\n\n"
        f"[NEW dataset_logic.py]\n```python\n{new_dataset_code}\n```\n"
        f"[CURRENT model.py]\n```python\n{current_model_code}\n```\n\n"
        f"[TASK] Rewrite `model.py` to implement the model architecture.\n"
        f"Ensure strictly less than {max_params} parameters.\n"
        f"Must be compatible with the NEW dataset logic."
    )
    model_response = await get_agent_response(
        "ModelArchitect", "", model_prompt, model_name
    )
    new_model_code = (
        extract_code_from_response(model_response, "ModelArchitect")
        or current_model_code
    )

    print(f"✅ [Phase 2-2] Model Logic Updated. Cooldown 60s...")
    await asyncio.sleep(60)

    print("\n[🌊 Pipeline] Step 3: Modifying Trainer Logic...")
    trainer_prompt = (
        f"{base_context}\n"
        f"{strategic_plan}\n\n"
        f"[NEW dataset_logic.py]\n```python\n{new_dataset_code}\n```\n"
        f"[NEW model.py]\n```python\n{new_model_code}\n```\n"
        f"[CURRENT trainer_logic.py]\n```python\n{current_trainer_code}\n```\n\n"
        f"[TASK] Rewrite `trainer_logic.py` to implement the training loop (loss, optimizer, scheduler).\n"
        f"Must integrate the NEW dataset and NEW model correctly."
    )
    trainer_response = await get_agent_response(
        "TrainerArchitect", "", trainer_prompt, model_name
    )
    new_trainer_code = (
        extract_code_from_response(trainer_response, "TrainerArchitect")
        or current_trainer_code
    )

    print(f"✅ [Phase 2-3] Trainer Logic Updated. Cooldown 60s...")
    await asyncio.sleep(60)

    generated_codes = {
        "dataset_logic.py": new_dataset_code,
        "model.py": new_model_code,
        "trainer_logic.py": new_trainer_code,
    }

    debate_log = {
        "timestamp": time.time(),
        "strategy": {"Theorist": theorist_msg, "Hacker": hacker_msg},
        "responses": {
            "DatasetArchitect": dataset_response,
            "ModelArchitect": model_response,
            "TrainerArchitect": trainer_response,
        },
    }

    if cycle_logger:
        try:
            cycle_logger.save_debate_log(debate_log)
            cycle_logger.save_debate_outputs(model_response, generated_codes)
            print("[System] Cycle logs saved successfully inside run_meeting.")
        except Exception as e:
            print(f"[Debate] ⚠️ Failed to save cycle logs: {e}")

    return generated_codes


def run_direct_training(exp_id, config):
    """Execute trainer.py directly to test the mutable trainer_logic.py."""
    log_file = os.path.join(LOG_DIR, f"exp_{exp_id}_train.log")
    print(
        f"[Executor] 🚀 Running training DIRECTLY via trainer.py (Log: {log_file})..."
    )
    capture_io = io.StringIO()
    keys_to_remove = [k for k in sys.modules if k.startswith("core_code")]
    for k in keys_to_remove:
        del sys.modules[k]

    status = "Success"
    error_msg = ""

    class DualOutput:
        def __init__(self, file_handle, stream_handle):
            self.file_handle = file_handle
            self.stream_handle = stream_handle

        def write(self, data):
            self.file_handle.write(data)
            self.stream_handle.write(data)
            sys.__stdout__.write(data)

        def flush(self):
            self.file_handle.flush()
            self.stream_handle.flush()
            sys.__stdout__.flush()

        def isatty(self):
            return sys.__stdout__.isatty()

    try:
        with open(log_file, "w", encoding="utf-8") as f:
            dual_out = DualOutput(f, capture_io)

            with redirect_stdout(dual_out), redirect_stderr(dual_out):
                try:
                    import core_code.trainer as trainer_module

                    if not hasattr(trainer_module, "main"):
                        raise ImportError(
                            "core_code.trainer.py must have a 'main(config)' function."
                        )

                    print(
                        f"[DirectRun] Loading fresh code from trainer.py & trainer_logic.py..."
                    )
                    trainer_module.main(config)

                    print(f"[DirectRun] Execution finished.")

                except Exception:
                    print("\n[DirectRun] 🚨 EXCEPTION CAUGHT!", file=sys.stderr)
                    traceback.print_exc(file=dual_out)
                    status = "Fail"
                    error_msg = "Runtime Exception"

    except Exception as e:
        status = "Fail"
        error_msg = str(e)
        capture_io.write(str(e))

    full_log = capture_io.getvalue()

    if status == "Fail" or "Traceback" in full_log or "Error" in full_log:
        print(f"\n[Executor] ⚠️ Execution failed. Preparing debug logs...")
        captured_log = full_log[-5000:] if len(full_log) > 5000 else full_log
        return {
            "status": "Fail",
            "error": error_msg,
            "trace": captured_log,
            "log_file": log_file,
        }

    try:
        from utils.log_parser import parse_log

        metrics = parse_log(log_file)
    except ImportError:
        metrics = {"status": "Success", "val_acc": 0.0}
    if "val_acc" in metrics:
        calc_eer = 100.0 - metrics.get("val_acc", 0.0)
        if metrics.get("eer") is None:
            metrics["eer"] = calc_eer

    try:
        log_content = full_log

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

        speed_matches = re.findall(r"\[Metrics\] Speed: ([\d\.]+)s/epoch", log_content)
        if speed_matches:
            speeds = [float(s) for s in speed_matches]
            metrics["speed"] = sum(speeds) / len(speeds)

    except Exception as e:
        print(f"[DirectRun] ⚠️ Log re-parsing failed: {e}")

    return metrics


def run_subprocess_training(exp_id, config):
    """
    Executes the training script and captures MAXIMUM logs for debugging.
    Improved: Forces traceback printing even if code tries to swallow errors via simple print.
    """
    import json
    import subprocess
    import sys
    import os
    from copy import deepcopy

    log_file = os.path.join(LOG_DIR, f"exp_{exp_id}_train.log")
    train_payload = deepcopy(config)
    train_config_str = json.dumps(train_payload)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    py_code = f"""
import sys
import traceback
import json
import os

def strict_excepthook(type, value, tb):
    print("!!! UNCAUGHT EXCEPTION IN SUBPROCESS !!!", file=sys.stderr)
    traceback.print_exception(type, value, tb)

sys.excepthook = strict_excepthook

try:
    sys.path.insert(0, '.')
    from core_code.trainer import main as run_trainer
    
    print("[Wrapper] Starting training...", flush=True)
    config = json.loads('''{train_config_str}''')
    run_trainer(config)
    print("[Wrapper] Training finished successfully.", flush=True)

except Exception as e:
    print("\\n[Wrapper] 🚨 Exception caught in wrapper!", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)
"""

    print(f"[Executor] 🚀 Running training subprocess (Log: {log_file})...")

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

            log_content = []

            while True:
                line = proc.stdout.readline()
                if not line and proc.poll() is not None:
                    break
                if line:
                    print(line, end="")
                    f.write(line)
                    log_content.append(line)

            remaining_stdout, _ = proc.communicate()
            if remaining_stdout:
                print(remaining_stdout, end="")
                f.write(remaining_stdout)
                log_content.append(remaining_stdout)

            ret = proc.returncode

            full_log_str = "".join(log_content)

            error_keywords = ["Traceback", "Error", "Exception", "Fail", "❌"]
            is_error = ret != 0 or any(k in full_log_str for k in error_keywords)

            if is_error:
                print(
                    f"\n[Executor] ⚠️ Process finished with issue (Code {ret}). Capturing extensive logs..."
                )

                if len(log_content) > 3000:
                    captured_log = "".join(log_content[-3000:])
                    captured_log = (
                        f"...(Truncated {len(log_content)-3000} lines)...\n"
                        + captured_log
                    )
                else:
                    captured_log = full_log_str

                return {
                    "status": "Fail",
                    "error": f"Error Detected (Code {ret})",
                    "trace": captured_log,
                    "log_file": log_file,
                }

    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"status": "Fail", "error": "Execution Logic Error", "trace": str(e)}

    metrics = parse_log(log_file)
    if metrics.get("status") == "Fail":
        return metrics

    if "val_acc" in metrics:
        metrics["eer"] = 100.0 - metrics.get("val_acc", 0.0)

    return metrics


async def execute_and_debug_cycle(exp_id, config, cycle_logger, max_debug_retries=10):
    """
    Main loop for Execution -> Check -> AutoDebug -> Retry
    """
    attempt = 0
    final_result = {"status": "Fail", "error": "Did not start"}

    if cycle_logger:
        trained_models_dir = os.path.join(
            cycle_logger.get_cycle_dir(), "trained_models"
        )
        config.setdefault("train", {})["checkpoint_dir"] = trained_models_dir

    while attempt <= max_debug_retries:
        print(f"\n⚡ [Execution Cycle] Attempt {attempt + 1} / {max_debug_retries + 1}")

        result = run_direct_training(exp_id, config)

        if result.get("status") == "Success":
            print(
                f"✅ [Success] Training completed. EER: {result.get('eer', 100):.2f}%"
            )
            return result

        print(f"❌ [Fail] {result.get('error')}")

        if attempt < max_debug_retries:
            print("[AutoDebug] 🚑 Initiating Multi-turn Debug Protocol...")

            current_codes = {}
            for filename in TARGET_FILES:
                path = os.path.join(CORE_CODE_DIR, filename)
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as f:
                        current_codes[filename] = f.read()

            full_log_trace = result.get("trace", result.get("error", "Unknown Error"))

            fixed_codes, debug_msg = await debug_failure(full_log_trace, current_codes)

            if fixed_codes:
                print(f"[AutoDebug] 🛠️ Applying fixes to {list(fixed_codes.keys())}...")
                for filename, code in fixed_codes.items():
                    if not code:
                        continue

                    src = os.path.join(CORE_CODE_DIR, filename)
                    if os.path.exists(src):
                        shutil.copy(src, f"{src}.bak_debug_{attempt}")

                    with open(src, "w", encoding="utf-8") as f:
                        f.write(code)

                if cycle_logger:
                    cycle_logger.save_debug_attempt(
                        attempt + 1,
                        {
                            "error": full_log_trace[-1000:],
                            "fix_files": list(fixed_codes.keys()),
                            "message": debug_msg,
                        },
                    )
            else:
                print("[AutoDebug] ⚠️ Debugger failed to generate fixes.")
                break

        final_result = result
        attempt += 1

    return final_result


async def debug_failure(
    error_trace: str, current_codes: Dict[str, str]
) -> Tuple[Dict[str, str], str]:
    """
    Multi-turn Debugging using 'chat_with_gem_response'.
    Flow: Analysis -> Dataset -> Model -> Trainer
    """
    target = ROLE_GEM_TARGETS.get("AutoDebugger")
    gem_id = target["id"] if target else None
    gem_name = target["names"][0] if target and target.get("names") else "Auto Debugger"
    if not gem_id:
        print("[AutoDebug] ❌ AutoDebugger Gem ID not found!")
        return {}, "Gem ID Missing"

    print(f"\n[🚑 AutoDebug] Starting Multi-turn Debug Session (Gem: {gem_id})...")
    code_context = ""
    for fname in ["dataset_logic.py", "model.py", "trainer_logic.py"]:
        content = current_codes.get(fname, "# Missing")
        code_context += f"\n<<<<FILE: {fname}>>>>\n{content}\n<<<<END: {fname}>>>>\n"
    print(f"error_trace: {error_trace}")
    prompt_1_analysis = (
        f"You are an expert Python Debugger.\n\n"
        f"[FULL EXECUTION LOG]\n{error_trace}\n\n"
        f"[CURRENT SOURCE CODE]\n{code_context}\n\n"
        f"Step 1. Analyze the error log carefully. Identify the root cause.\n"
        f"Step 2. Outline which files need to be changed and why.\n"
        f"DO NOT output code yet. Just explain the bug and the fix plan."
    )

    print("[AutoDebug] 💬 Turn 1: Analyzing Error & Planning...")
    analysis_resp, session_id = chat_with_gem_response(
        prompt_1_analysis, gem_id=gem_id, session_id=None, gem_name=gem_name
    )
    print(f"[AutoDebug] 🧠 Analysis: {analysis_resp[:200]}...")

    fixed_codes = {}
    prompt_2_dataset = (
        "Now, based on your plan, please provide the corrected `dataset_logic.py`.\n"
        "Even if there are no changes, output the full valid code to ensure consistency.\n"
        "Output ONLY the code inside a standard Python code block (```python ... ```)."
    )

    print("[AutoDebug] 💬 Turn 2: Requesting dataset_logic.py...")
    ds_resp, _ = chat_with_gem_response(
        prompt_2_dataset, gem_id=gem_id, session_id=session_id
    )
    fixed_codes["dataset_logic.py"] = extract_code_block(ds_resp)
    prompt_3_model = (
        "Next, provide the corrected `model.py`.\n"
        "Ensure it is compatible with the `dataset_logic.py` you just provided.\n"
        "Output ONLY the code inside a standard Python code block."
    )

    print("[AutoDebug] 💬 Turn 3: Requesting model.py...")
    model_resp, _ = chat_with_gem_response(
        prompt_3_model, gem_id=gem_id, session_id=session_id
    )
    fixed_codes["model.py"] = extract_code_block(model_resp)
    prompt_4_trainer = (
        "Finally, provide the corrected `trainer_logic.py`.\n"
        "Ensure it correctly uses the model and dataset from previous turns.\n"
        "Output ONLY the code inside a standard Python code block."
    )

    print("[AutoDebug] 💬 Turn 4: Requesting trainer_logic.py...")
    trainer_resp, _ = chat_with_gem_response(
        prompt_4_trainer, gem_id=gem_id, session_id=session_id
    )
    fixed_codes["trainer_logic.py"] = extract_code_block(trainer_resp)
    if not all(fixed_codes.values()):
        print("[AutoDebug] ⚠️ Some files were not returned correctly.")

    return fixed_codes, "Multi-turn Debug Completed"


async def summarize_experiment(
    cycle_id: int,
    metrics: Dict[str, Any],
    old_codes: Dict[str, str],
    new_codes: Dict[str, str],
    debate_log: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Uses the ChangeSummarizer agent to generate a concise summary of the cycle.
    """
    print(f"\n[Summarizer] 📝 Generating LLM summary for Cycle {cycle_id}...")

    strategy_text = "No strategy recorded."
    if debate_log and "strategy" in debate_log:
        s = debate_log["strategy"]
        strategy_text = f"Theorist: {s.get('Theorist', '')[:300]}...\nHacker: {s.get('Hacker', '')[:300]}..."

    changes_text = ""
    for filename in new_codes:
        old_c = old_codes.get(filename, "")
        new_c = new_codes.get(filename, "")
        if old_c != new_c:
            changes_text += (
                f"\n[File: {filename} (Modified)]\n```python\n{new_c}\n```\n"
            )

    prompt = (
        f"Analyze the changes in Cycle {cycle_id} and provide a concise summary for the experiment history.\n\n"
        f"[METRICS RESULT]\n"
        f"Status: {metrics.get('status')}\n"
        f"EER: {metrics.get('eer', 'N/A')}%\n"
        f"Val Acc: {metrics.get('val_acc', 'N/A')}%\n\n"
        f"[STRATEGY HYPOTHESIS]\n{strategy_text}\n\n"
        f"[MODIFIED CODE]\n{changes_text}\n\n"
        f"[TASK]\n"
        f"1. Briefly explain the Core Hypothesis (what was attempted).\n"
        f"2. Summarize the Key Technical Changes (architectural or logic changes).\n"
        f"3. Correlate changes with the result (Success/Fail/Improvement).\n"
        f"4. Output must be a single paragraph, concise, under 300 characters only text if possible."
    )
    system_instruction = (
        "You are an expert Machine Learning Researcher and Technical Writer. "
        "Your task is to analyze code changes and experiment metrics to create a concise, "
        "high-density summary of a specific experimental cycle."
    )
    print(f"{prompt}")
    summary = await generate_gemini_response(
        user_prompt=prompt,
        system_instruction=system_instruction,
        model="gemini-3.0-pro",
    )
    if not summary:
        print("[Summarizer] ⚠️ Summary generation failed.")
        return "Summary generation failed."

    return summary
