"""Sequential RGB decoding with explicitly owned and closed FFmpeg pipes."""
from __future__ import annotations

import re
import subprocess
import tempfile


def read_rgb_frames(path):
    from imageio_ffmpeg import get_ffmpeg_exe
    executable = get_ffmpeg_exe()
    probe = subprocess.run([executable, '-hide_banner', '-noautorotate', '-i', str(path),
        '-map', '0:v:0', '-frames:v', '0', '-f', 'null', '-'], capture_output=True, text=True, timeout=30)
    if probe.returncode:
        raise ValueError(f'FFmpeg could not inspect video: {probe.stderr[-2000:]}')
    stream = next((line for line in probe.stderr.splitlines()
                   if 'Stream #' in line and 'Video:' in line), '')
    size = re.search(r'\b(\d{2,5})x(\d{2,5})(?:\s|,|\[)', stream)
    rate = re.search(r'([\d.]+)\s+fps\b', stream)
    if not size or not rate:
        raise ValueError('FFmpeg video dimensions or FPS are missing')
    width, height = map(int, size.groups())
    fps = float(rate.group(1))
    yield {'size': (width, height), 'fps': fps}
    frame_bytes = width*height*3
    with tempfile.TemporaryFile() as diagnostics:
        with subprocess.Popen([executable, '-v', 'error', '-noautorotate', '-i', str(path),
                '-map', '0:v:0', '-an', '-sn', '-vsync', '0', '-f', 'rawvideo',
                '-pix_fmt', 'rgb24', 'pipe:1'], stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=diagnostics) as process:
            try:
                while True:
                    frame = process.stdout.read(frame_bytes)
                    if not frame:
                        break
                    if len(frame) != frame_bytes:
                        raise ValueError('FFmpeg produced a partial RGB frame')
                    yield frame
                if process.wait(timeout=30):
                    diagnostics.seek(0)
                    raise ValueError(diagnostics.read().decode('utf-8', errors='replace')[-2000:])
            finally:
                process.stdout.close()
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
