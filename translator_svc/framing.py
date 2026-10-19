"""Binary frame encoding for the StreamingResponse wire protocol.

Wire format per frame:
    [1 byte: type] [4 bytes: length, big-endian] [length bytes: payload]
"""

import json
from enum import IntEnum


class FrameType(IntEnum):
    SOURCE = 0x01
    TARGET = 0x02
    AUDIO = 0x03
    ERROR = 0xFF


_LENGTH_BYTES = 4
_MAX_PAYLOAD_BYTES = 2**32 - 1


class FrameEncoder:
    def encode(self, frame_type: FrameType, payload: bytes) -> bytes:
        if len(payload) > _MAX_PAYLOAD_BYTES:
            raise ValueError(
                f"Frame payload {len(payload)} bytes exceeds {_MAX_PAYLOAD_BYTES}"
            )
        return (
            bytes([int(frame_type)])
            + len(payload).to_bytes(_LENGTH_BYTES, "big")
            + payload
        )

    def encode_error(self, code: str, message: str) -> bytes:
        payload = json.dumps({"code": code, "message": message}).encode("utf-8")
        return self.encode(FrameType.ERROR, payload)
