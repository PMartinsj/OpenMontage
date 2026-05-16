"""1min.ai text-to-speech provider tool.

Calls the 1min.ai unified feature API (POST /api/features) with
type=TEXT_TO_SPEECH. Supports OpenAI-compatible voices (alloy, echo,
fable, onyx, nova, shimmer) via the tts-1 and tts-1-hd models.

Auth: API-KEY header (not Bearer token).
"""

from __future__ import annotations

import os
import time
import uuid  # used for output filename fallback
from pathlib import Path
from typing import Any

import requests

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)

_BASE_URL = "https://api.1min.ai/api/features"
_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
_MODELS = ["tts-1", "tts-1-hd"]


class OneMinAITTS(BaseTool):
    name = "oneminai_tts"
    version = "1.0.0"
    tier = ToolTier.VOICE
    capability = "tts"
    provider = "1minai"
    stability = ToolStability.BETA
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.API

    dependencies = ["requests"]
    install_instructions = (
        "Set the ONEMINAI_API_KEY environment variable:\n"
        "  export ONEMINAI_API_KEY=your_key_here\n"
        "Get a key at https://1min.ai"
    )
    fallback = "openai_tts"
    fallback_tools = ["openai_tts", "piper_tts"]
    agent_skills = ["text-to-speech"]

    capabilities = ["text_to_speech", "voice_selection"]
    supports = {
        "voice_cloning": False,
        "multilingual": True,
        "offline": False,
        "native_audio": True,
    }
    best_for = [
        "OpenAI-quality TTS via 1min.ai subscription",
        "German narration with tts-1-hd",
        "multi-voice dialogue (alloy/nova/shimmer)",
    ]
    not_good_for = [
        "voice cloning",
        "offline production",
    ]

    # Recommended voice mapping for German dialogue
    VOICE_ROLES = {
        "narrator": "nova",    # warm, professional female
        "lena": "shimmer",     # softer, younger-sounding female
        "sabine": "alloy",     # calm, neutral female
    }

    input_schema = {
        "type": "object",
        "required": ["text"],
        "properties": {
            "text": {"type": "string", "description": "Text to synthesise (max 4096 chars)"},
            "voice": {
                "type": "string",
                "default": "nova",
                "enum": _VOICES,
                "description": "Voice name. Use VOICE_ROLES mapping for dialogue.",
            },
            "model": {
                "type": "string",
                "default": "tts-1-hd",
                "enum": _MODELS,
                "description": "tts-1-hd for highest quality, tts-1 for faster/cheaper.",
            },
            "speed": {
                "type": "number",
                "default": 1.0,
                "description": "Speech speed (0.25–4.0).",
            },
            "format": {
                "type": "string",
                "default": "mp3",
                "enum": ["mp3", "opus", "aac", "flac", "wav", "pcm"],
            },
            "output_path": {
                "type": "string",
                "description": "Where to save the audio file.",
            },
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=64, vram_mb=0, disk_mb=10, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=2, retryable_errors=["rate_limit", "timeout"])
    idempotency_key_fields = ["text", "voice", "model", "format"]
    side_effects = ["writes audio file to output_path", "calls 1min.ai API"]
    user_visible_verification = ["Listen to generated audio for intelligibility and tone"]

    def get_status(self) -> ToolStatus:
        if os.environ.get("ONEMINAI_API_KEY"):
            return ToolStatus.AVAILABLE
        return ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        # 1min.ai charges via credits; approximate at OpenAI tts-1-hd rate
        chars = len(inputs.get("text", ""))
        return round(chars * 0.000030, 4)

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        api_key = os.environ.get("ONEMINAI_API_KEY")
        if not api_key:
            return ToolResult(success=False, error="ONEMINAI_API_KEY not set. " + self.install_instructions)

        start = time.time()
        try:
            result = self._generate(inputs, api_key)
        except Exception as exc:
            return ToolResult(success=False, error=f"1min.ai TTS failed: {exc}")

        result.duration_seconds = round(time.time() - start, 2)
        result.cost_usd = self.estimate_cost(inputs)
        return result

    def _generate(self, inputs: dict[str, Any], api_key: str) -> ToolResult:
        from tools.analysis.audio_probe import probe_duration

        text = inputs["text"]
        voice = inputs.get("voice", "nova")
        model = inputs.get("model", "tts-1-hd")
        fmt = inputs.get("format", "mp3")
        speed = inputs.get("speed", 1.0)
        output_path = Path(inputs.get("output_path", f"oneminai_tts_{uuid.uuid4().hex[:8]}.{fmt}"))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "type": "TEXT_TO_SPEECH",
            "model": model,
            "promptObject": {
                "text": text,
                "voice": voice,
                "response_format": fmt,
            },
        }
        if speed != 1.0:
            payload["promptObject"]["speed"] = speed

        resp = requests.post(
            _BASE_URL,
            json=payload,
            headers={
                "API-KEY": api_key,
                "Content-Type": "application/json",
            },
            timeout=60,
        )
        resp.raise_for_status()

        body = resp.json()
        audio_url = self._extract_audio_url(body)
        if not audio_url:
            return ToolResult(
                success=False,
                error=f"No audio URL in 1min.ai response: {body}",
            )

        # Download the audio file from the temporary URL
        audio_resp = requests.get(audio_url, timeout=60)
        audio_resp.raise_for_status()
        output_path.write_bytes(audio_resp.content)

        audio_duration = probe_duration(output_path)

        return ToolResult(
            success=True,
            data={
                "provider": self.provider,
                "model": model,
                "voice": voice,
                "format": fmt,
                "text_length": len(text),
                "audio_duration_seconds": round(audio_duration, 2) if audio_duration else None,
                "output": str(output_path),
            },
            artifacts=[str(output_path)],
            model=model,
        )

    @staticmethod
    def _extract_audio_url(body: dict) -> str | None:
        """Pull the temporaryUrl from the 1min.ai response envelope.

        Actual response shape: body["aiRecord"]["temporaryUrl"]
        """
        ai_record = body.get("aiRecord", {})
        if url := ai_record.get("temporaryUrl"):
            return url
        # Fallback: top-level (future-proofing)
        if url := body.get("temporaryUrl"):
            return url
        return None
