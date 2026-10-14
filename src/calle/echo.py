"""The step-2 "brain": repeat back whatever the user said.

Step 3 replaces this with an LLM; everything around it stays the same.
"""

from pipecat.frames.frames import (
    Frame,
    TranscriptionFrame,
    TTSSpeakFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class EchoProcessor(FrameProcessor):
    """Turns the user's finished utterance into a TTSSpeakFrame.

    Why not echo every TranscriptionFrame immediately? Deepgram can emit
    several *final* transcripts for one long utterance (one per phrase), and
    some arrive while the user is still talking. So we buffer finals and
    speak only once both are true:

      1. the VAD says the user has stopped speaking, and
      2. we have at least one final transcript.

    Either event can happen first (Deepgram is sometimes faster than the VAD,
    sometimes slower), so we check after each. Known limitation: if Deepgram
    finalizes a phrase *before* the VAD stop and another *after*, the reply is
    split in two. Step 5 swaps this for proper turn detection.
    """

    def __init__(self, prefix: str = "You said: ") -> None:
        super().__init__()
        self._prefix = prefix
        self._buffer: list[str] = []
        self._user_speaking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        # Required: lets the base class handle lifecycle frames (start, end...).
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._user_speaking = True
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._user_speaking = False
            await self._maybe_speak()
        elif isinstance(frame, TranscriptionFrame):
            self._buffer.append(frame.text.strip())
            await self._maybe_speak()

        # Always pass every frame along, even ones we used. Downstream stages
        # (and the browser client's transcript view) may need them; the TTS
        # service knows to ignore transcription frames rather than speak them.
        await self.push_frame(frame, direction)

    async def _maybe_speak(self) -> None:
        if self._user_speaking or not self._buffer:
            return
        text = " ".join(part for part in self._buffer if part)
        self._buffer.clear()
        if text:
            await self.push_frame(TTSSpeakFrame(f"{self._prefix}{text}"))
