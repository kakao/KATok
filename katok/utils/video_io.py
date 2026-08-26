"""Reading and writing videos.

Uses PyAV, so any container ffmpeg can open works. Tensors follow the model's
convention: ``(C, T, H, W)`` float in ``[-1, 1]``.
"""

from fractions import Fraction

import numpy as np
import torch


def read_video(path: str, n_frames: int | None = None, stride: int = 1, start: int = 0) -> torch.Tensor:
    """Decode a video into a ``(C, T, H, W)`` float tensor in ``[0, 1]``.

    Args:
        path: video file to read.
        n_frames: stop after this many sampled frames. ``None`` reads to the end.
        stride: keep every ``stride``-th frame.
        start: skip this many frames before sampling.
    """
    import av

    frames = []
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for i, frame in enumerate(container.decode(video=0)):
            if i < start or (i - start) % stride:
                continue
            frames.append(frame.to_ndarray(format="rgb24"))
            if n_frames is not None and len(frames) == n_frames:
                break

    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    if n_frames is not None and len(frames) < n_frames:
        raise RuntimeError(f"{path}: requested {n_frames} frames but only {len(frames)} available")

    video = torch.from_numpy(np.stack(frames))  # (T, H, W, C) uint8
    return video.permute(3, 0, 1, 2).float() / 255.0


def write_video(video: torch.Tensor, path: str, fps: float = 24.0, crf: int = 18) -> None:
    """Write a ``(C, T, H, W)`` tensor in ``[-1, 1]`` to an H.264 mp4.

    Values are clamped, so out-of-range reconstructions are saturated rather than
    wrapped. ``crf`` is the x264 quality knob; lower is better and 18 is close to
    visually lossless.
    """
    import av

    if video.ndim != 4:
        raise ValueError(f"expected (C, T, H, W), got {tuple(video.shape)}")

    frames = ((video.detach().float().cpu().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    frames = frames.permute(1, 2, 3, 0).numpy()  # (T, H, W, C)
    _, H, W, _ = frames.shape

    if H % 2 or W % 2:
        raise ValueError(f"H and W must be even for yuv420p, got {H}x{W}")

    with av.open(path, mode="w") as container:
        # PyAV wants an exact rational frame rate, not a float.
        stream = container.add_stream("libx264", rate=Fraction(fps).limit_denominator(1000))
        stream.width, stream.height = W, H
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf)}

        for frame in frames:
            av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            container.mux(stream.encode(av_frame))
        container.mux(stream.encode())


def save_frames_grid(video: torch.Tensor, path: str, every: int = 1) -> None:
    """Write frames of a ``(C, T, H, W)`` tensor in ``[-1, 1]`` side by side as a PNG."""
    from PIL import Image

    frames = ((video.detach().float().cpu().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    frames = frames[:, ::every].permute(1, 2, 3, 0).numpy()  # (T, H, W, C)
    Image.fromarray(np.concatenate(list(frames), axis=1)).save(path)
