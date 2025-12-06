import asyncio
import sys
import os
import time
import json
import random
import uuid
from pathlib import Path
from typing import List, Union, Dict, Optional, Tuple
from datetime import datetime
import nest_asyncio
from gemini_webapi import GeminiClient, ChatSession, Gem

nest_asyncio.apply()

GEMINI_COOKIE = {"Secure_1PSID": "Put Yours", "Secure_1PSIDTS": "Put Yours"}
_client: GeminiClient = None
_chat_sessions: Dict[str, ChatSession] = {}


def _reset_client_cache():
    """Reset client to force reconnection on next request"""
    global _client, _chat_sessions
    if _client:
        print("[Gemini] Clearing client cache", file=sys.stderr)
    _client = None
    _chat_sessions.clear()


async def _get_client():
    """Singleton Client Management: Cookie -> Fallback to Auto"""
    global _client
    if _client is not None:
        return _client

    if GEMINI_COOKIE:
        try:
            print(
                "[Gemini-WebAPI] 🍪 Attempting login with provided cookies...",
                file=sys.stderr,
            )
            client = GeminiClient(
                GEMINI_COOKIE["Secure_1PSID"],
                GEMINI_COOKIE["Secure_1PSIDTS"],
                proxy=None,
            )
            await client.init(timeout=600, auto_close=True, auto_refresh=True)

            _client = client
            print("[Gemini-WebAPI] ✓ Cookie login successful", file=sys.stderr)
            return _client
        except Exception as e:
            print(
                f"[Gemini-WebAPI] ⚠️ Cookie login failed: {e}. Falling back to auto-login...",
                file=sys.stderr,
            )

    try:
        print(
            "[Gemini-WebAPI] 🌐 Initializing with auto-browser login...",
            file=sys.stderr,
        )
        client = GeminiClient(proxy=None)
        await client.init(timeout=600, auto_close=True, auto_refresh=True)

        _client = client
        print("[Gemini-WebAPI] ✓ Client initialized successfully", file=sys.stderr)
    except Exception as e:
        print(
            f"[ERROR] All login methods failed: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        _client = None

    return _client


async def _generate_content_async(
    prompt: str, model: str, files: List[Union[str, Path]] = None
):
    """Internal async function for single turn content generation using Library"""
    try:
        print(
            f"[DEBUG] generate_content start - model: {model}, prompt_len: {len(prompt)}",
            file=sys.stderr,
        )
        client = await _get_client()
        if not client:
            raise RuntimeError("Client Init Failed")
        print(f"[Summarizer] ⏳ Cooldown complete. Generating summary...")
        await asyncio.sleep(60)
        response = await client.generate_content(prompt, model=model, files=files)
        print(
            f"[DEBUG] ✓ Response received - len: {len(response.text) if response.text else 0}",
            file=sys.stderr,
        )
        print(f"[Summarizer] ✅ Summary generation complete. 60s cooldown...")
        await asyncio.sleep(60)
        return response.text

    except Exception as e:
        error_str = str(e)
        print(f"[ERROR] {type(e).__name__}: {error_str}", file=sys.stderr)
        raise


async def generate_gemini_response(
    user_prompt: str,
    system_instruction: str = None,
    files: List[Union[str, Path]] = None,
    model: str = "gemini-3.0-pro",
) -> str:
    """Single turn query wrapper"""
    final_prompt = user_prompt
    if system_instruction:
        final_prompt = (
            f"System Instruction:\n{system_instruction}\n\nUser Query:\n{user_prompt}"
        )
    loop = asyncio.get_event_loop()
    return _execute_with_retry(
        loop, _generate_content_async, final_prompt, model, files
    )


async def _send_chat_message_async(
    session_id: str, prompt: str, model: str, files: List[Union[str, Path]] = None
):
    """Internal async function for chat message"""
    try:
        print(
            f"[DEBUG] _send_chat_message_async start - session: {session_id}, model: {model}",
            file=sys.stderr,
        )
        global _chat_sessions
        client = await _get_client()
        if not client:
            raise RuntimeError("Client Init Failed")

        if session_id not in _chat_sessions:
            print(f"[Gemini] Creating new chat session: {session_id}", file=sys.stderr)
            _chat_sessions[session_id] = client.start_chat(model=model)

        chat = _chat_sessions[session_id]
        response = await chat.send_message(prompt, files=files)
        print(
            f"[DEBUG] send_message completed - response_len: {len(response.text) if response.text else 0}",
            file=sys.stderr,
        )
        await asyncio.sleep(3)
        return response.text
    except Exception as e:
        print(
            f"[ERROR] _send_chat_message_async error: {type(e).__name__}: {str(e)}",
            file=sys.stderr,
        )
        raise


def chat_gemini_response(
    user_prompt: str,
    session_id: str = None,
    files: List[Union[str, Path]] = None,
    model: str = "gemini-3.0-pro",
) -> Tuple[str, str]:
    """
    Multi-turn chat wrapper.
    """
    loop = asyncio.get_event_loop()

    if not session_id:
        session_id = str(uuid.uuid4())[:8]

    response_text = _execute_with_retry(
        loop, _send_chat_message_async, session_id, user_prompt, model, files
    )
    return response_text, session_id


async def fetch_gems(include_hidden: bool = False) -> Dict[str, Gem]:
    """Fetch available gems"""
    client = await _get_client()
    if not client:
        raise RuntimeError("Failed to initialize Gemini client.")
    return await client.fetch_gems(include_hidden=include_hidden)


async def create_gem(name: str, prompt: str, description: str = "") -> Gem:
    """Create a new custom gem"""
    client = await _get_client()
    if not client:
        raise RuntimeError("Failed to initialize Gemini client.")
    return await client.create_gem(name=name, prompt=prompt, description=description)


async def delete_gem(gem_id: str) -> None:
    """Delete a gem by ID"""
    client = await _get_client()
    if not client:
        raise RuntimeError("Failed to initialize Gemini client.")
    await client.delete_gem(gem=gem_id)
    print(f"Gem with ID {gem_id} has been deleted.")


async def get_gem_by_id(gem_id: str) -> Optional[Gem]:
    """Retrieve a gem by ID"""
    client = await _get_client()
    if not client:
        raise RuntimeError("Failed to initialize Gemini client.")
    gems = await client.fetch_gems(include_hidden=True)
    return gems.get(gem_id) if gems else None


async def get_gem_by_name(name: str) -> Optional[Gem]:
    """Retrieve a gem by Name"""
    client = await _get_client()
    if not client:
        raise RuntimeError("Failed to initialize Gemini client.")
    gems = await client.fetch_gems(include_hidden=True)
    if gems:
        for gem in gems.values():
            if gem.name == name:
                return gem
    return None


async def apply_gem_and_request(gem_id: str, user_prompt: str) -> str:
    """Apply a specific gem for a single request"""
    client = await _get_client()
    if not client:
        raise RuntimeError("Failed to initialize Gemini client.")

    gem = await get_gem_by_id(gem_id)
    if not gem:
        raise ValueError(f"Gem with ID {gem_id} not found.")

    response = await client.generate_content(
        prompt=user_prompt, model="gemini-3.0-pro", gem=gem
    )
    return response.text


async def generate_with_gem(
    user_prompt: str, gem: Optional[Gem] = None, model: str = "gemini-3.0-pro"
) -> str:
    """Generate content using a specific gem object"""
    client = await _get_client()
    if not client:
        raise RuntimeError("Failed to initialize Gemini client.")

    if not gem:
        raise ValueError("A valid gem must be provided.")

    response = await client.generate_content(prompt=user_prompt, model=model, gem=gem)
    return response.text


async def _send_gem_chat_async(
    session_id: str,
    prompt: str,
    gem_id: str,
    files: List[Union[str, Path]] = None,
    gem_name: str = None,
):
    """Gem Chat async internal function"""
    try:
        global _chat_sessions
        client = await _get_client()
        if not client:
            raise RuntimeError("Client Init Failed")

        if session_id not in _chat_sessions:
            if not gem_id:
                raise ValueError("gem_id is required to start a new session.")

            gem = None
            retry_count = 0
            max_retries = 5
            while not gem:
                gem = await get_gem_by_id(gem_id)
                if not gem:
                    gem = await get_gem_by_name(gem_name)
                    print("found by name")
                if gem:
                    print(
                        f"[Gemini] ✓ Gem '{gem.name}' found (ID: {gem.id})",
                        file=sys.stderr,
                    )
                    break

                retry_count += 1
                if retry_count > max_retries:
                    raise ValueError(
                        f"Gem '{gem_id}' not found after {max_retries} retries."
                    )

                print(
                    f"[Gemini] ⚠️ Gem '{gem_id}' not found. Retrying in 3s... ({retry_count}/{max_retries})",
                    file=sys.stderr,
                )
                await asyncio.sleep(3)
                client = await _get_client()

            print(
                f"[Gemini] Starting Chat with Gem '{gem.name}': {session_id}",
                file=sys.stderr,
            )
            _chat_sessions[session_id] = client.start_chat(gem=gem)

        chat = _chat_sessions[session_id]
        print(f"[DEBUG] Sending Gem Chat (session: {session_id})", file=sys.stderr)
        response = await chat.send_message(prompt, files=files)
        print(
            f"[DEBUG] Gem Response Received - len: {len(response.text) if response.text else 0}",
            file=sys.stderr,
        )
        await asyncio.sleep(2)
        return response.text

    except Exception as e:
        print(
            f"[ERROR] _send_gem_chat_async error: {type(e).__name__}: {str(e)}",
            file=sys.stderr,
        )
        raise


def chat_with_gem_response(
    user_prompt: str,
    gem_id: str,
    session_id: str = None,
    files: List[Union[str, Path]] = None,
    gem_name: str = None,
) -> Tuple[str, str]:
    """Gem Multi-turn Wrapper"""
    loop = asyncio.get_event_loop()

    if not session_id:
        session_id = str(uuid.uuid4())[:8]

    response_text = _execute_with_retry(
        loop, _send_gem_chat_async, session_id, user_prompt, gem_id, files, gem_name
    )
    return response_text, session_id


def _execute_with_retry(loop, async_func, *args):
    """Common infinite retry logic wrapper"""
    attempt = 0

    while True:
        attempt += 1
        try:
            print(f"[DEBUG] Attempt {attempt} - {async_func.__name__}", file=sys.stderr)
            result = loop.run_until_complete(async_func(*args))
            print(f"[DEBUG] ✓✓✓ Success (Attempt {attempt}) ✓✓✓", file=sys.stderr)
            return result

        except Exception as e:
            error_str = str(e)
            print(
                f"\n[ERROR] Failed (Attempt {attempt}) - {type(e).__name__}",
                file=sys.stderr,
            )
            print(f"[ERROR] Message: {error_str}", file=sys.stderr)

            if "401" in error_str or "403" in error_str:
                print("[ERROR] Auth failed! Check cookies.", file=sys.stderr)
            _reset_client_cache()
            base_delay = min(60, 2 * (1.5 ** (attempt - 1)))
            wait_time = base_delay + random.uniform(0, 1)
            print(f"[Gemini] Retrying in {wait_time:.2f}s...\n", file=sys.stderr)
            time.sleep(wait_time)


def log_llm_interaction(
    role: str,
    system_instruction: str,
    user_prompt: str,
    response: str,
    used_real: bool,
    model_name: str = "gemini-3.0-pro",
    cycle_logger=None,
) -> None:
    """Log LLM interaction"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "role": role,
        "model": model_name,
        "api_mode": "web_api",
        "system_instruction": system_instruction if system_instruction else None,
        "user_prompt_length": len(user_prompt),
        "response_length": len(response) if response else 0,
    }

    if cycle_logger:
        try:
            qna_file = os.path.join(cycle_logger.dirs["debate"], "qna_log.jsonl")
            with open(qna_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            return
        except Exception as e:
            print(f"Failed to log interaction via CycleLogger: {e}", file=sys.stderr)


def log_qna_session(
    session_id: str,
    role: str,
    turns: List[Dict[str, str]],
    model_name: str = "gemini-3.0-pro",
    cycle_logger=None,
) -> None:
    """Log entire QnA session"""
    session_log = {
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "role": role,
        "model": model_name,
        "api_mode": "web_api",
        "turn_count": len(turns),
        "turns": turns,
    }

    if cycle_logger:
        try:
            session_file = os.path.join(
                cycle_logger.dirs["debate"], f"session_{session_id}.json"
            )
            with open(session_file, "w", encoding="utf-8") as f:
                json.dump(session_log, f, indent=2, ensure_ascii=False)
            return
        except Exception as e:
            print(f"Failed to log session via CycleLogger: {e}", file=sys.stderr)

    os.makedirs("logs", exist_ok=True)
    session_file = f"logs/gemini_session_{session_id}.json"
    try:
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(session_log, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Failed to log session: {e}", file=sys.stderr)
