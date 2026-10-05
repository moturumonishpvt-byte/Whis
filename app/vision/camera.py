"""Camera capture utilities for WHIS.

Provides EXPLICIT, on-demand camera frame capture.

NO continuous camera monitoring.
NO background camera access.
All camera access is a single explicit API call.

Uses ctypes/Win32 where possible. If a camera is unavailable,
capture returns a CaptureResult with success=False and a clear
error message. No exception is raised to the caller.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from app.vision.screen import CaptureResult

logger = logging.getLogger(__name__)


def capture_camera_frame(
    device_index: int = 0,
    output_path: Optional[str] = None,
) -> CaptureResult:
    """Capture a single frame from a connected camera device.

    This is an EXPLICIT, on-demand call. There is no continuous, background,
    or always-on camera monitoring.

    Uses PowerShell + Windows.Media.Capture where available. If no camera
    is accessible or the capture fails, returns CaptureResult(success=False).

    Args:
        device_index: Camera device index (0 = default/first camera).
        output_path: File path for the captured frame. Defaults to a temp file.

    Returns:
        CaptureResult with success flag, image path, and dimensions.

    Note:
        Requires a physical camera device. Returns success=False if no camera
        is connected or access is denied.
    """
    try:
        if output_path is None:
            tmp = tempfile.NamedTemporaryFile(suffix=".bmp", delete=False)
            tmp.close()
            output_path = tmp.name
            Path(output_path).unlink(missing_ok=True)

        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        dest_str = str(dest).replace("'", "''")

        ps_script = f"""
Add-Type @"
using System;
using System.Runtime.InteropServices;

public class CamCapture {{
    [DllImport("avicap32.dll")]
    public static extern IntPtr capCreateCaptureWindowA(
        string lpszWindowName, int dwStyle, int x, int y,
        int nWidth, int nHeight, IntPtr hwndParent, int nID);

    [DllImport("user32.dll")]
    public static extern bool SendMessage(IntPtr hWnd, uint Msg, IntPtr wParam, IntPtr lParam);

    [DllImport("user32.dll")]
    public static extern bool DestroyWindow(IntPtr hWnd);
}}
"@

$output = '{dest_str}'
$hWnd = [CamCapture]::capCreateCaptureWindowA("cam", 0x10000000, 0, 0, 320, 240, [IntPtr]::Zero, 0)
if ($hWnd -eq [IntPtr]::Zero) {{
    Write-Error "Failed to create capture window."
    exit 1
}}
$connected = [CamCapture]::SendMessage($hWnd, 0x040A, [IntPtr]{device_index}, [IntPtr]::Zero)
if (-not $connected) {{
    [CamCapture]::DestroyWindow($hWnd) | Out-Null
    [Console]::Error.WriteLine("No camera device connected.")
    exit 1
}}
Start-Sleep -Milliseconds 500
[CamCapture]::SendMessage($hWnd, 0x0419, [IntPtr]::Zero, [IntPtr]::Zero) | Out-Null
[CamCapture]::SendMessage($hWnd, 0x041C, [System.IntPtr]::Zero, [System.Runtime.InteropServices.Marshal]::StringToHGlobalAnsi($output)) | Out-Null
[CamCapture]::SendMessage($hWnd, 0x040B, [IntPtr]{device_index}, [IntPtr]::Zero) | Out-Null
[CamCapture]::DestroyWindow($hWnd) | Out-Null
if ((Test-Path $output) -and ((Get-Item $output).Length -gt 0)) {{ exit 0 }} else {{ [Console]::Error.WriteLine("Camera frame capture failed."); exit 1 }}
"""

        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0 and dest.exists() and dest.stat().st_size > 0:
            logger.info(
                "Camera frame captured from device %d to %s.", device_index, dest
            )
            return CaptureResult(
                success=True,
                image_path=str(dest),
                width=320,
                height=240,
            )
        else:
            error_detail = result.stderr.strip() or result.stdout.strip() or "Camera not available."
            logger.warning("Camera capture failed: %s", error_detail)
            return CaptureResult(
                success=False,
                error=f"Camera capture failed: {error_detail}",
            )

    except subprocess.TimeoutExpired:
        return CaptureResult(
            success=False,
            error="Camera capture timed out after 10s.",
        )
    except Exception as exc:
        logger.error("capture_camera_frame raised unexpected error: %s", exc)
        return CaptureResult(success=False, error=str(exc))
