"""Straico image generation tool.

Calls POST https://api.straico.com/v1/image/generation.
Supports 20+ models: openai/gpt-image-1, openai/dall-e-3, flux/1.1,
ideogram/V_2A, fal-ai/recraft/v3/text-to-image, fal-ai/ideogram/v3, etc.

Auth: uses OPENAI_API_KEY (the Straico key, which starts with "OT-") as
Bearer token. OPENAI_BASE_URL must point to https://api.straico.com/v2.

Request body: { model, description, size, variations }
  - description: the image prompt (NOT "prompt")
  - size: "square" | "landscape" | "portrait" (model-dependent variants exist)
  - variations: number of images (int, min 1)

Response: { data: { images: [url, ...], zip: url, price: { total: coins } }, success: true }
  - images are direct S3 download URLs (temporary, download immediately)
  - price in Straico coins (not USD); 1 coin ≈ $0.001
"""

from __future__ import annotations

import os
import time
import uuid
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

_API_URL = "https://api.straico.com/v1/image/generation"
_MODELS_URL = "https://api.straico.com/v1/models"

# Verified available models and their coin costs per square image (May 2026)
MODELS = {
    "openai/gpt-image-1": 37,        # cheapest, good quality
    "openai/dall-e-3": 90,
    "fal-ai/ideogram/v3": 100,
    "ideogram/V_1_TURBO": 90,
    "ideogram/V_2A_TURBO": 113,
    "fal-ai/recraft/v3/text-to-image": 133,
    "ideogram/V_2A": 180,
    "fal-ai/imagen4/preview": 167,
    "flux/1.1": 250,
    "FLUX-1.1-pro": 250,
    "fal-ai/flux/dev": 250,
}

DEFAULT_MODEL = "openai/gpt-image-1"


class StraicoImage(BaseTool):
    name = "straico_image"
    version = "1.0.0"
    tier = ToolTier.GENERATE
    capability = "image_generation"
    provider = "straico"
    stability = ToolStability.BETA
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.API

    dependencies = ["requests"]
    install_instructions = (
        "Set OPENAI_API_KEY to your Straico API key (starts with OT-).\n"
        "Set OPENAI_BASE_URL=https://api.straico.com/v2.\n"
        "Get a key at https://straico.com"
    )
    fallback = "openai_image"
    fallback_tools = ["openai_image", "flux_image"]
    agent_skills = ["flux-best-practices"]

    capabilities = ["generate_image", "generate_illustration", "text_to_image"]
    supports = {
        "complex_instructions": True,
        "text_in_image": True,
        "multiple_outputs": True,
        "offline": False,
        "multilingual_prompts": True,
    }
    best_for = [
        "budget image gen via Straico subscription (gpt-image-1 at 37 coins)",
        "photorealistic scenes (Flux 1.1, Ideogram V2A)",
        "infographics and diagrams with text (gpt-image-1, Ideogram)",
        "consistent style across multiple images",
    ]
    not_good_for = [
        "offline production",
        "voice cloning",
    ]

    input_schema = {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Image description / prompt (max ~4000 chars)",
            },
            "model": {
                "type": "string",
                "default": DEFAULT_MODEL,
                "enum": list(MODELS.keys()),
                "description": "Straico image model ID. Default: openai/gpt-image-1 (cheapest, 37 coins/image).",
            },
            "size": {
                "type": "string",
                "default": "square",
                "enum": ["square", "landscape", "portrait"],
                "description": "'square' (1024x1024), 'landscape' (1792x1024 equiv), 'portrait' (1024x1792 equiv).",
            },
            "variations": {
                "type": "integer",
                "default": 1,
                "minimum": 1,
                "maximum": 4,
                "description": "Number of images to generate.",
            },
            "output_path": {
                "type": "string",
                "description": "Where to save the image (PNG). Defaults to a temp name.",
            },
            "output_dir": {
                "type": "string",
                "description": "Directory for output when variations > 1.",
            },
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=256, vram_mb=0, disk_mb=20, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=2, retryable_errors=["rate_limit", "timeout"])
    idempotency_key_fields = ["prompt", "model", "size"]
    side_effects = ["writes image file(s) to output_path", "calls Straico API (costs coins)"]
    user_visible_verification = ["Inspect generated image for relevance and quality"]

    def _api_key(self) -> str | None:
        # Straico uses the OPENAI_API_KEY slot (their key starts with "OT-")
        key = os.environ.get("OPENAI_API_KEY", "")
        if key and key.startswith("OT-"):
            return key
        # Also check explicit STRAICO_API_KEY
        return os.environ.get("STRAICO_API_KEY") or None

    def get_status(self) -> ToolStatus:
        return ToolStatus.AVAILABLE if self._api_key() else ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        model = inputs.get("model", DEFAULT_MODEL)
        n = inputs.get("variations", 1)
        coins_per_image = MODELS.get(model, 200)
        total_coins = coins_per_image * n
        # 1 Straico coin ≈ $0.001 (rough estimate; varies by subscription tier)
        return round(total_coins * 0.001, 4)

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        api_key = self._api_key()
        if not api_key:
            return ToolResult(
                success=False,
                error="Straico API key not found. " + self.install_instructions,
            )

        start = time.time()
        try:
            result = self._generate(inputs, api_key)
        except Exception as exc:
            return ToolResult(success=False, error=f"Straico image generation failed: {exc}")

        result.duration_seconds = round(time.time() - start, 2)
        result.cost_usd = self.estimate_cost(inputs)
        return result

    def _generate(self, inputs: dict[str, Any], api_key: str) -> ToolResult:
        prompt = inputs["prompt"]
        model = inputs.get("model", DEFAULT_MODEL)
        size = inputs.get("size", "square")
        variations = inputs.get("variations", 1)

        payload = {
            "model": model,
            "description": prompt,
            "size": size,
            "variations": variations,
        }

        resp = requests.post(
            _API_URL,
            json=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=120,
        )
        resp.raise_for_status()

        body = resp.json()
        if not body.get("success"):
            return ToolResult(
                success=False,
                error=f"Straico API error: {body.get('error', body)}",
            )

        image_urls: list[str] = body["data"]["images"]
        coins_used: int = body["data"].get("price", {}).get("total", 0)

        # Download all images
        saved_paths: list[str] = []
        for i, url in enumerate(image_urls):
            if variations == 1 and inputs.get("output_path"):
                out = Path(inputs["output_path"])
            elif inputs.get("output_dir"):
                out = Path(inputs["output_dir"]) / f"image_{i+1}.png"
            else:
                slug = uuid.uuid4().hex[:8]
                out = Path(f"straico_{slug}_{i+1}.png")

            out.parent.mkdir(parents=True, exist_ok=True)
            img_resp = requests.get(url, timeout=60)
            img_resp.raise_for_status()
            out.write_bytes(img_resp.content)
            saved_paths.append(str(out))

        return ToolResult(
            success=True,
            data={
                "provider": "straico",
                "model": model,
                "size": size,
                "variations": variations,
                "coins_used": coins_used,
                "outputs": saved_paths,
                "output": saved_paths[0] if saved_paths else None,
            },
            artifacts=saved_paths,
            model=model,
        )
