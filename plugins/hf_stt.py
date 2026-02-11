"""
Hugging Face STT plugin for LiveKit agents.
Runs locally in-process (no API): uses transformers (ASR pipeline) with models from the Hugging Face Hub.
Requires: pip install transformers torch
"""
import asyncio
import logging
from typing import Optional

import numpy as np
import transformers  # Hugging Face model library; pipeline() used in _transcribe_sync

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    stt,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer
from livekit import rtc

logger = logging.getLogger(__name__)


def _transcribe_sync(model_id: str, audio_bytes: bytes, sample_rate: int) -> str:
    """Run HF ASR in a thread; returns transcribed text."""
    pipe = transformers.pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device=-1,
    )
    # pipeline expects {"array": ndarray (float), "sampling_rate": int}
    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    audio_float = audio_int16.astype(np.float32) / 32768.0
    out = pipe({"array": audio_float, "sampling_rate": sample_rate})
    if isinstance(out, dict):
        return out.get("text", out.get("chunks", [{}])[0].get("text", "") if out.get("chunks") else "") or ""
    return str(out)


class HFSTTSpeechStream(stt.SpeechStream):
    """RecognizeStream that buffers audio, runs Hugging Face ASR on flush, and emits SpeechEvents."""

    def __init__(
        self,
        *,
        stt_instance: "HFSTT",
        conn_options: APIConnectOptions,
        sample_rate: Optional[int] = None,
        model_id: str = "",
    ) -> None:
        super().__init__(
            stt=stt_instance,
            conn_options=conn_options,
            sample_rate=sample_rate if sample_rate is not None else NOT_GIVEN,
        )
        self._model_id = model_id

    async def _run(self) -> None:
        # RecognizeStream sends _FlushSentinel to mark end of segment
        def is_flush(item):  # noqa: E306
            return type(item).__name__ == "_FlushSentinel"
        buffer: list[bytes] = []
        sr = 0
        loop = asyncio.get_event_loop()
        try:
            async for item in self._input_ch:
                if is_flush(item):
                    if not buffer or sr <= 0:
                        continue
                    pcm = b"".join(buffer)
                    buffer = []
                    request_id = utils.shortuuid()
                    try:
                        text = await loop.run_in_executor(
                            None,
                            _transcribe_sync,
                            self._model_id,
                            pcm,
                            sr,
                        )
                    except Exception as exc:
                        logger.exception("HF STT failed: %s", exc)
                        raise APIConnectionError() from exc
                    duration_sec = len(pcm) / (2 * sr) if sr else 0  # 16-bit = 2 bytes per sample
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(
                            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                            request_id=request_id,
                            alternatives=[stt.SpeechData(language="en", text=text.strip() or "")],
                            recognition_usage=stt.RecognitionUsage(audio_duration=duration_sec),
                        )
                    )
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(
                            type=stt.SpeechEventType.RECOGNITION_USAGE,
                            request_id=request_id,
                            recognition_usage=stt.RecognitionUsage(audio_duration=duration_sec),
                        )
                    )
                else:
                    assert isinstance(item, rtc.AudioFrame)
                    buffer.append(bytes(item.data))
                    if sr <= 0:
                        sr = item.sample_rate
        except Exception as exc:
            logger.exception("HF STT stream error: %s", exc)
            raise


class HFSTT(stt.STT):
    """STT using Hugging Face transformers (automatic-speech-recognition pipeline)."""

    def __init__(
        self,
        model: str = "openai/whisper-tiny",
    ) -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=True, interim_results=False))
        self._model_id = model

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        frames = buffer if isinstance(buffer, list) else [buffer]
        if not frames:
            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                request_id=utils.shortuuid(),
                alternatives=[stt.SpeechData(language="en", text="")],
                recognition_usage=stt.RecognitionUsage(audio_duration=0.0),
            )
        sr = frames[0].sample_rate
        pcm = b"".join(bytes(f.data) for f in frames)
        loop = asyncio.get_event_loop()
        try:
            text = await loop.run_in_executor(
                None,
                _transcribe_sync,
                self._model_id,
                pcm,
                sr,
            )
        except Exception as exc:
            logger.exception("HF STT recognize failed: %s", exc)
            from livekit.agents._exceptions import APIConnectionError
            raise APIConnectionError() from exc
        duration_sec = sum(f.duration for f in frames)
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            request_id=utils.shortuuid(),
            alternatives=[stt.SpeechData(language="en", text=text.strip() or "")],
            recognition_usage=stt.RecognitionUsage(audio_duration=duration_sec),
        )

    def stream(
        self,
        *,
        language: Optional[str] = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        sample_rate: Optional[int] = None,
    ) -> HFSTTSpeechStream:
        return HFSTTSpeechStream(
            stt_instance=self,
            conn_options=conn_options,
            sample_rate=sample_rate,
            model_id=self._model_id,
        )
