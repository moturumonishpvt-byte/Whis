"""Screen capture utilities for WHIS.

Provides explicit, on-demand screen and region capture using native
Windows GDI calls (ctypes + user32/gdi32). No third-party dependencies.
No continuous or background capture.

All captures are explicit API calls only.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Windows GDI constants
SRCCOPY = 0x00CC0020
DIB_RGB_COLORS = 0
BI_RGB = 0


@dataclass(frozen=True)
class ScreenRegion:
    """Defines a rectangular region of the screen in pixels."""

    left: int
    top: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"Region dimensions must be positive. Got width={self.width}, height={self.height}."
            )


@dataclass(frozen=True)
class CaptureResult:
    """Result of a screen capture operation."""

    success: bool
    image_path: Optional[str] = None
    width: int = 0
    height: int = 0
    error: Optional[str] = None


def _get_screen_dimensions() -> Tuple[int, int]:
    """Return the primary monitor's (width, height) in pixels."""
    user32 = ctypes.windll.user32
    # SM_CXSCREEN = 0, SM_CYSCREEN = 1
    width = user32.GetSystemMetrics(0)
    height = user32.GetSystemMetrics(1)
    return width, height


def _capture_region_to_bmp(
    left: int,
    top: int,
    width: int,
    height: int,
    output_path: Path,
) -> None:
    """Capture a screen region and save as a 24-bit BMP using Win32 GDI.

    Uses BitBlt to copy from the screen DC into a memory DC, then
    extracts raw pixel data and writes a valid BMP file without any
    third-party libraries.

    Args:
        left: Left edge of capture region in screen coordinates.
        top: Top edge of capture region in screen coordinates.
        width: Width of capture region in pixels.
        height: Height of capture region in pixels.
        output_path: Destination file path for the BMP image.

    Raises:
        OSError: If GDI operations fail or the file cannot be written.
        ValueError: If dimensions are invalid.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid capture dimensions: {width}x{height}")

    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32

    # Get screen DC
    hdc_screen = user32.GetDC(None)
    if not hdc_screen:
        raise OSError("GetDC failed: cannot obtain screen device context.")

    # Create compatible memory DC and bitmap
    hdc_mem = gdi32.CreateCompatibleDC(hdc_screen)
    hbmp = gdi32.CreateCompatibleBitmap(hdc_screen, width, height)
    old_obj = gdi32.SelectObject(hdc_mem, hbmp)

    try:
        # BitBlt: copy from screen to memory
        result = gdi32.BitBlt(hdc_mem, 0, 0, width, height, hdc_screen, left, top, SRCCOPY)
        if not result:
            raise OSError("BitBlt failed: screen capture operation returned 0.")

        # Extract pixel data via GetDIBits
        # BITMAPINFOHEADER: biSize, biWidth, biHeight (negative = top-down), biPlanes,
        #                   biBitCount, biCompression, biSizeImage, ... (all zeros)
        bmi_size = 40
        row_bytes = ((width * 24 + 31) // 32) * 4  # DWORD-aligned
        image_size = row_bytes * height
        bmi = struct.pack(
            "<IIIHHIIIIII",
            bmi_size,  # biSize
            width,     # biWidth
            -height,   # biHeight (negative = top-down)
            1,         # biPlanes
            24,        # biBitCount
            BI_RGB,    # biCompression
            image_size,# biSizeImage
            0, 0,      # biXPelsPerMeter, biYPelsPerMeter
            0, 0,      # biClrUsed, biClrImportant
        )

        buf = ctypes.create_string_buffer(image_size)
        lines_copied = gdi32.GetDIBits(
            hdc_mem, hbmp, 0, height,
            buf,
            ctypes.create_string_buffer(bmi, len(bmi)),
            DIB_RGB_COLORS,
        )
        if lines_copied <= 0:
            raise OSError(f"GetDIBits failed: returned {lines_copied} scan lines.")

        # Write BMP file
        file_size = 14 + bmi_size + image_size
        pixel_offset = 14 + bmi_size
        bmp_header = struct.pack(
            "<2sIHHI",
            b"BM",        # signature
            file_size,    # file size
            0, 0,         # reserved
            pixel_offset, # pixel data offset
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(bmp_header)
            f.write(bmi)
            f.write(buf.raw)

        logger.debug("Captured %dx%d region to %s.", width, height, output_path)

    finally:
        gdi32.SelectObject(hdc_mem, old_obj)
        gdi32.DeleteObject(hbmp)
        gdi32.DeleteDC(hdc_mem)
        user32.ReleaseDC(None, hdc_screen)


def capture_screen(output_path: Optional[str] = None) -> CaptureResult:
    """Capture the full primary monitor screen.

    This is an EXPLICIT, on-demand capture. There is no continuous or
    background capture mechanism.

    Args:
        output_path: Destination file path. Defaults to a temp file if not given.

    Returns:
        CaptureResult with success flag, image path, and dimensions.
    """
    try:
        width, height = _get_screen_dimensions()
        if width <= 0 or height <= 0:
            return CaptureResult(
                success=False,
                error=f"Invalid screen dimensions returned: {width}x{height}.",
            )

        if output_path is None:
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix=".bmp", delete=False)
            tmp.close()
            output_path = tmp.name

        dest = Path(output_path)
        _capture_region_to_bmp(0, 0, width, height, dest)

        return CaptureResult(
            success=True,
            image_path=str(dest),
            width=width,
            height=height,
        )
    except Exception as exc:
        logger.error("capture_screen failed: %s", exc)
        return CaptureResult(success=False, error=str(exc))


def capture_region(
    left: int,
    top: int,
    width: int,
    height: int,
    output_path: Optional[str] = None,
) -> CaptureResult:
    """Capture a rectangular region of the screen.

    This is an EXPLICIT, on-demand capture. No continuous monitoring.

    Args:
        left: Left edge in screen coordinates.
        top: Top edge in screen coordinates.
        width: Width of capture region in pixels.
        height: Height of capture region in pixels.
        output_path: Destination file path. Defaults to a temp file.

    Returns:
        CaptureResult with success flag, image path, and dimensions.
    """
    try:
        if width <= 0 or height <= 0:
            return CaptureResult(
                success=False,
                error=f"Invalid region dimensions: width={width}, height={height}.",
            )

        if output_path is None:
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix=".bmp", delete=False)
            tmp.close()
            output_path = tmp.name

        dest = Path(output_path)
        _capture_region_to_bmp(left, top, width, height, dest)

        return CaptureResult(
            success=True,
            image_path=str(dest),
            width=width,
            height=height,
        )
    except Exception as exc:
        logger.error("capture_region failed: %s", exc)
        return CaptureResult(success=False, error=str(exc))
