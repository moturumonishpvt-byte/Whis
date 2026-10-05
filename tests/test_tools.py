"""Unit and integration tests for WHIS Stage 6: Tools and Windows Control."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from app.core.permissions import PermissionDecision, PermissionLevel, PermissionManager
from app.tools import (
    BaseTool,
    CreateDirectoryTool,
    DeleteFileTool,
    ExecuteCommandTool,
    FunctionalTool,
    GetCurrentTimeTool,
    GetRunningApplicationsTool,
    GetSystemInfoTool,
    LaunchApplicationTool,
    ListFilesTool,
    MoveFileTool,
    OpenURLTool,
    ReadFileTool,
    ToolRegistry,
    ToolResult,
    WriteFileTool,
    create_default_registry,
)


class TestToolsSubsystem(unittest.TestCase):
    """Test suite covering tools, permissions, file safety, and application launching."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.tmp_dir.name)
        self.registry = create_default_registry()

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_01_tool_registration(self) -> None:
        """1. Custom tool registers and appears in list_tools()."""
        custom_tool = FunctionalTool(
            name="test_tool",
            description="A test tool",
            func=lambda: "result",
            permission_level=PermissionLevel.SAFE,
        )
        self.registry.register(custom_tool)
        self.assertIsNotNone(self.registry.get("test_tool"))
        names = [t["name"] for t in self.registry.list_tools()]
        self.assertIn("test_tool", names)

    def test_02_tool_lookup(self) -> None:
        """2. Registry returns correct tool instance by name."""
        tool = self.registry.get("get_current_time")
        self.assertIsNotNone(tool)
        self.assertEqual(tool.name, "get_current_time")
        self.assertEqual(tool.permission_level, PermissionLevel.SAFE)

    def test_03_unknown_tool_execution(self) -> None:
        """3. Executing an unknown tool returns a failed ToolResult without raising exception."""
        res = self.registry.execute("unknown_tool_xyz")
        self.assertIsInstance(res, ToolResult)
        self.assertFalse(res.success)
        self.assertIn("not found", res.error)

    def test_04_safe_tool_execution_without_confirmation(self) -> None:
        """4. SAFE tool executes immediately without confirmation."""
        res = self.registry.execute("get_current_time", confirmed=False)
        self.assertTrue(res.success)
        self.assertIn("local_time", res.output)
        self.assertFalse(res.requires_confirmation)

    def test_05_confirm_tool_without_confirmation_blocked(self) -> None:
        """5. CONFIRM tool is blocked with requires_confirmation=True when confirmed=False."""
        target_file = str(self.test_dir / "unconfirmed.txt")
        res = self.registry.execute("write_file", confirmed=False, path=target_file, content="Blocked")
        self.assertFalse(res.success)
        self.assertTrue(res.requires_confirmation)
        self.assertFalse(Path(target_file).exists())

    def test_06_confirm_tool_with_confirmation_allowed(self) -> None:
        """6. CONFIRM tool executes successfully when confirmed=True."""
        target_file = str(self.test_dir / "confirmed.txt")
        res = self.registry.execute("write_file", confirmed=True, path=target_file, content="Allowed")
        self.assertTrue(res.success)
        self.assertTrue(Path(target_file).exists())
        self.assertEqual(Path(target_file).read_text(encoding="utf-8"), "Allowed")

    def test_07_high_risk_tool_blocked_without_confirmation(self) -> None:
        """7. HIGH_RISK tool is blocked when confirmed=False."""
        dummy_file = self.test_dir / "to_delete.txt"
        dummy_file.write_text("content", encoding="utf-8")

        res = self.registry.execute("delete_file", confirmed=False, path=str(dummy_file))
        self.assertFalse(res.success)
        self.assertTrue(res.requires_confirmation)
        self.assertTrue(dummy_file.exists())

    def test_08_high_risk_tool_allowed_with_confirmation(self) -> None:
        """8. HIGH_RISK tool executes when confirmed=True."""
        dummy_file = self.test_dir / "to_delete.txt"
        dummy_file.write_text("content", encoding="utf-8")

        res = self.registry.execute("delete_file", confirmed=True, path=str(dummy_file))
        self.assertTrue(res.success)
        self.assertFalse(dummy_file.exists())

    def test_09_system_info_tool(self) -> None:
        """9. get_system_info returns structured OS, CPU, and RAM data."""
        res = self.registry.execute("get_system_info")
        self.assertTrue(res.success)
        data = res.output
        self.assertIn("os", data)
        self.assertIn("cpu_count_logical", data)
        self.assertIn("ram_total_gb", data)
        self.assertGreater(data["ram_total_gb"], 0)

    def test_10_running_applications_tool(self) -> None:
        """10. get_running_applications returns list of processes."""
        res = self.registry.execute("get_running_applications", limit=10)
        self.assertTrue(res.success)
        self.assertIsInstance(res.output, list)
        if res.output:
            self.assertIn("name", res.output[0])
            self.assertIn("pid", res.output[0])

    def test_11_directory_creation(self) -> None:
        """11. create_directory creates requested directory structure."""
        sub_dir = str(self.test_dir / "folder_a" / "folder_b")
        res = self.registry.execute("create_directory", confirmed=True, path=sub_dir)
        self.assertTrue(res.success)
        self.assertTrue(Path(sub_dir).is_dir())

    def test_12_file_writing(self) -> None:
        """12. write_file writes file content and parent directories."""
        fpath = str(self.test_dir / "nested" / "output.txt")
        res = self.registry.execute("write_file", confirmed=True, path=fpath, content="Hello WHIS")
        self.assertTrue(res.success)
        self.assertEqual(Path(fpath).read_text(encoding="utf-8"), "Hello WHIS")

    def test_13_file_reading(self) -> None:
        """13. read_file reads text file and handles truncation metadata."""
        fpath = self.test_dir / "sample.txt"
        fpath.write_text("Sample file content for test", encoding="utf-8")

        res = self.registry.execute("read_file", path=str(fpath))
        self.assertTrue(res.success)
        self.assertEqual(res.output["content"], "Sample file content for test")
        self.assertFalse(res.output["truncated"])

    def test_14_file_listing(self) -> None:
        """14. list_files lists directory entries with size and metadata."""
        (self.test_dir / "file1.txt").write_text("123", encoding="utf-8")
        (self.test_dir / "subdir").mkdir()

        res = self.registry.execute("list_files", path=str(self.test_dir))
        self.assertTrue(res.success)
        item_names = [i["name"] for i in res.output["items"]]
        self.assertIn("file1.txt", item_names)
        self.assertIn("subdir", item_names)

    def test_15_file_moving(self) -> None:
        """15. move_file renames or moves a file."""
        src = self.test_dir / "source.txt"
        dst = self.test_dir / "destination.txt"
        src.write_text("move me", encoding="utf-8")

        res = self.registry.execute("move_file", confirmed=True, source=str(src), destination=str(dst))
        self.assertTrue(res.success)
        self.assertFalse(src.exists())
        self.assertTrue(dst.exists())
        self.assertEqual(dst.read_text(encoding="utf-8"), "move me")

    def test_16_file_deletion_protection(self) -> None:
        """16. delete_file rejects directory targets (no recursive destructive deletion)."""
        sub = self.test_dir / "protected_dir"
        sub.mkdir()

        res = self.registry.execute("delete_file", confirmed=True, path=str(sub))
        self.assertFalse(res.success)
        self.assertIn("directory", res.error)
        self.assertTrue(sub.exists())

    def test_17_file_deletion_success(self) -> None:
        """17. delete_file deletes individual file with explicit confirmation."""
        f = self.test_dir / "temp_file.txt"
        f.write_text("trash", encoding="utf-8")

        res = self.registry.execute("delete_file", confirmed=True, path=str(f))
        self.assertTrue(res.success)
        self.assertFalse(f.exists())

    def test_18_path_validation_and_null_bytes(self) -> None:
        """18. Path containing null bytes is rejected."""
        res = self.registry.execute("read_file", path="file\x00name.txt")
        self.assertFalse(res.success)
        self.assertIn("Null bytes", res.error)

    def test_19_empty_path_rejection(self) -> None:
        """19. Empty or whitespace paths are rejected."""
        res = self.registry.execute("read_file", path="   ")
        self.assertFalse(res.success)
        self.assertIn("cannot be empty", res.error)

    def test_20_application_launch_validation(self) -> None:
        """20. launch_application rejects forbidden shell metacharacters."""
        res = self.registry.execute("launch_application", confirmed=True, name_or_path="notepad.exe & calc.exe")
        self.assertFalse(res.success)
        self.assertIn("metacharacters", res.error)

    @patch("subprocess.Popen")
    def test_21_application_launch_executable_resolution(self, mock_popen: MagicMock) -> None:
        """21. launch_application maps common app name and launches with shell=False."""
        mock_proc = MagicMock()
        mock_proc.pid = 9999
        mock_popen.return_value = mock_proc

        res = self.registry.execute("launch_application", confirmed=True, name_or_path="notepad")
        self.assertTrue(res.success)
        self.assertEqual(res.output["pid"], 9999)

        mock_popen.assert_called_once()
        call_kwargs = mock_popen.call_args[1]
        self.assertFalse(call_kwargs.get("shell", True))

    def test_22_url_validation_allowed_schemes(self) -> None:
        """22. open_url permits valid http and https URLs."""
        with patch("webbrowser.open", return_value=True):
            res = self.registry.execute("open_url", url="https://github.com/ggml-org/llama.cpp")
            self.assertTrue(res.success)
            self.assertEqual(res.output["url"], "https://github.com/ggml-org/llama.cpp")

    def test_23_url_validation_rejected_schemes(self) -> None:
        """23. open_url rejects non-http schemes like file:// or javascript:."""
        res = self.registry.execute("open_url", url="file:///C:/Windows/System32/cmd.exe")
        self.assertFalse(res.success)
        self.assertIn("scheme", res.error)

        res_js = self.registry.execute("open_url", url="javascript:alert(1)")
        self.assertFalse(res_js.success)
        self.assertIn("scheme", res_js.error)

    def test_24_command_tool_high_risk_enforcement(self) -> None:
        """24. execute_command is classified as HIGH_RISK and blocked without confirmation."""
        tool = self.registry.get("execute_command")
        self.assertEqual(tool.permission_level, PermissionLevel.HIGH_RISK)

        res = self.registry.execute("execute_command", confirmed=False, command=["cmd.exe", "/c", "echo test"])
        self.assertFalse(res.success)
        self.assertTrue(res.requires_confirmation)

    @patch("subprocess.run")
    def test_25_command_tool_shell_false(self, mock_run: MagicMock) -> None:
        """25. execute_command runs with discrete argument list and strictly shell=False."""
        mock_res = MagicMock(returncode=0, stdout="hello", stderr="")
        mock_run.return_value = mock_res

        res = self.registry.execute(
            "execute_command",
            confirmed=True,
            command=["echo", "hello"],
        )
        self.assertTrue(res.success)
        self.assertEqual(res.output["stdout"], "hello")

        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        self.assertEqual(kwargs.get("shell"), False)
        self.assertIsInstance(args[0], list)


if __name__ == "__main__":
    unittest.main()
