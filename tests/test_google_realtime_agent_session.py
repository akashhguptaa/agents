from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

import pytest
from google.genai import types
from google.genai.live import AsyncLive

from livekit.agents import Agent, AgentSession, SpeechCreatedEvent, llm
from livekit.plugins import google
from livekit.plugins.google.realtime.realtime_api import RealtimeSession

from .fake_io import FakeAudioOutput

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _livekit_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIVEKIT_API_KEY", "k")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "s")


def _manual_activity_config() -> types.RealtimeInputConfig:
    return types.RealtimeInputConfig(
        automatic_activity_detection=types.AutomaticActivityDetection(disabled=True)
    )


class _ScriptedFakeLiveSession:
    def __init__(self) -> None:
        self.content_calls: list[tuple[object, bool]] = []
        self.tool_response_calls: list[object] = []
        self.realtime_input_calls: list[dict[str, object]] = []
        self._queue: asyncio.Queue[types.LiveServerMessage | None] = asyncio.Queue()

    async def send_client_content(self, *, turns: object, turn_complete: bool) -> None:
        self.content_calls.append((turns, turn_complete))

    async def send_tool_response(self, *, function_responses: object) -> None:
        self.tool_response_calls.append(function_responses)

    async def send_realtime_input(self, **kwargs: object) -> None:
        self.realtime_input_calls.append(kwargs)

    async def receive(self) -> AsyncIterator[types.LiveServerMessage]:
        while True:
            msg = await self._queue.get()
            if msg is None:
                return
            yield msg

    async def close(self) -> None:
        self._queue.put_nowait(None)

    def push(self, msg: types.LiveServerMessage) -> None:
        self._queue.put_nowait(msg)

    @property
    def activity_starts(self) -> list[dict[str, object]]:
        return [c for c in self.realtime_input_calls if "activity_start" in c]


@asynccontextmanager
async def _agent_session(
    monkeypatch: pytest.MonkeyPatch,
    *,
    manual_activity: bool,
) -> AsyncIterator[tuple[AgentSession, RealtimeSession, _ScriptedFakeLiveSession]]:
    fake = _ScriptedFakeLiveSession()

    @asynccontextmanager
    async def _connect(
        self: AsyncLive, **kwargs: object
    ) -> AsyncIterator[_ScriptedFakeLiveSession]:
        yield fake

    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    monkeypatch.setattr(AsyncLive, "connect", _connect)

    model_kwargs: dict[str, object] = {}
    if manual_activity:
        model_kwargs["realtime_input_config"] = _manual_activity_config()

    session = AgentSession(
        llm=google.realtime.RealtimeModel(**model_kwargs),
        vad=None,
        turn_handling={"turn_detection": "manual"},
    )
    session.output.audio = FakeAudioOutput()
    await session.start(Agent(instructions="test"))

    activity = session._activity
    assert activity is not None
    rt_session = activity._rt_session
    assert isinstance(rt_session, RealtimeSession)

    async def _wait_connected() -> None:
        while rt_session._active_session is None:
            await asyncio.sleep(0.01)

    await asyncio.wait_for(_wait_connected(), timeout=5.0)
    await asyncio.sleep(0.05)

    try:
        yield session, rt_session, fake
    finally:
        await session.aclose()


async def _wait_until(predicate: Callable[[], bool], *, timeout: float) -> None:
    async def _poll() -> None:
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(_poll(), timeout=timeout)


async def test_manual_transcript_only_turn_unblocks_the_next_generate_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _agent_session(monkeypatch, manual_activity=True) as (session, rt_session, fake):
        speech_created: list[SpeechCreatedEvent] = []
        session.on("speech_created", speech_created.append)
        transcripts: list[llm.InputTranscriptionCompleted] = []
        rt_session.on("input_audio_transcription_completed", transcripts.append)

        fake.push(
            types.LiveServerMessage(
                server_content=types.LiveServerContent(
                    input_transcription=types.Transcription(text="Hello?")
                )
            )
        )
        fake.push(
            types.LiveServerMessage(
                server_content=types.LiveServerContent(
                    input_transcription=types.Transcription(text="", finished=True)
                )
            )
        )

        await _wait_until(lambda: any(ev.is_final for ev in transcripts), timeout=1.0)

        session.generate_reply(instructions="x")

        await _wait_until(
            lambda: any(turn_complete for _, turn_complete in fake.content_calls),
            timeout=1.0,
        )

        assert [ev.user_initiated for ev in speech_created] == [True]


async def test_manual_activity_model_turn_does_not_reach_the_socket_as_activity_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _agent_session(monkeypatch, manual_activity=True) as (session, rt_session, fake):
        started: list[llm.InputSpeechStartedEvent] = []
        rt_session.on("input_speech_started", started.append)
        created: list[llm.GenerationCreatedEvent] = []
        rt_session.on("generation_created", created.append)

        fake.push(
            types.LiveServerMessage(
                server_content=types.LiveServerContent(
                    model_turn=types.Content(parts=[types.Part(text="Thank you for calling.")])
                )
            )
        )

        await _wait_until(lambda: len(created) == 1, timeout=1.0)
        await asyncio.sleep(0.2)

        assert started == []
        assert fake.activity_starts == []


async def test_automatic_activity_model_turn_still_reaches_the_socket_as_activity_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _agent_session(monkeypatch, manual_activity=False) as (session, rt_session, fake):
        started: list[llm.InputSpeechStartedEvent] = []
        rt_session.on("input_speech_started", started.append)

        fake.push(
            types.LiveServerMessage(
                server_content=types.LiveServerContent(
                    model_turn=types.Content(parts=[types.Part(text="Hello")])
                )
            )
        )

        await _wait_until(lambda: len(started) == 1, timeout=1.0)
