"""Tests for the Codex-backed compatibility integration layer."""

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.claude.exceptions import ClaudeProcessError, ClaudeTimeoutError
from src.claude.sdk_integration import ClaudeResponse, ClaudeSDKManager, StreamUpdate
from src.config.settings import Settings


class _Stream:
    """Async readline stream backed by a list of byte lines."""

    def __init__(self, lines=None, delay: float = 0.0):
        self._lines = list(lines or [])
        self._delay = delay

    async def readline(self) -> bytes:
        if self._delay:
            await asyncio.sleep(self._delay)
        if not self._lines:
            return b""
        return self._lines.pop(0)


class _MockProcess:
    """Minimal asyncio subprocess-compatible mock."""

    def __init__(self, stdout_lines=None, stderr_lines=None, returncode=0, delay=0.0):
        self.stdout = _Stream(stdout_lines, delay=delay)
        self.stderr = _Stream(stderr_lines, delay=delay)
        self.returncode = returncode

    async def wait(self) -> int:
        return self.returncode

    def kill(self) -> None:
        self.returncode = -9


@pytest.fixture
def config(tmp_path: Path) -> Settings:
    return Settings(
        telegram_bot_token="test:token",
        telegram_bot_username="testbot",
        approved_directory=tmp_path,
        claude_timeout_seconds=2,
    )


@pytest.fixture
def manager(config: Settings) -> ClaudeSDKManager:
    with patch("src.claude.sdk_integration.find_codex_cli", return_value="/usr/bin/codex"):
        return ClaudeSDKManager(config)


class TestClaudeSDKManager:
    async def test_execute_command_success(self, manager: ClaudeSDKManager):
        async def _create_process(*cmd, **kwargs):
            cmd_list = list(cmd)
            output_index = cmd_list.index("--output-last-message") + 1
            output_path = Path(cmd_list[output_index])
            output_path.write_text("final message from codex", encoding="utf-8")

            stdout_lines = [
                b'{"type":"thread.started","thread_id":"thread-123"}\n',
                b'{"type":"turn.started"}\n',
                b'{"type":"response.output_text.delta","delta":"hello"}\n',
                b'{"type":"exec.command.started","command":"ls -la"}\n',
            ]
            return _MockProcess(stdout_lines=stdout_lines, returncode=0)

        with patch(
            "src.claude.sdk_integration.asyncio.create_subprocess_exec",
            side_effect=_create_process,
        ):
            response = await manager.execute_command(
                prompt="say hello",
                working_directory=Path("/tmp"),
            )

        assert isinstance(response, ClaudeResponse)
        assert response.session_id == "thread-123"
        assert response.content == "final message from codex"
        assert response.num_turns == 1
        assert response.duration_ms >= 0
        assert response.cost == 0.0
        assert any(tool.get("name") == "Bash" for tool in response.tools_used)

    async def test_execute_command_resume_session(self, manager: ClaudeSDKManager):
        called_cmd = []

        async def _create_process(*cmd, **kwargs):
            cmd_list = list(cmd)
            called_cmd.extend(cmd_list)

            return _MockProcess(
                stdout_lines=[
                    b'{"type":"turn.started"}\n',
                    b'{"type":"response.output_text.delta","delta":"continued"}\n',
                ],
                returncode=0,
            )

        with patch(
            "src.claude.sdk_integration.asyncio.create_subprocess_exec",
            side_effect=_create_process,
        ):
            response = await manager.execute_command(
                prompt="continue",
                working_directory=Path("/tmp"),
                session_id="thread-existing",
                continue_session=True,
            )

        # codex exec resume --json --skip-git-repo-check <session_id>
        assert called_cmd[0:3] == ["/usr/bin/codex", "exec", "resume"]
        assert "--json" in called_cmd
        assert "--skip-git-repo-check" in called_cmd
        assert called_cmd.index("--json") < called_cmd.index("thread-existing")
        assert called_cmd.index("--skip-git-repo-check") < called_cmd.index(
            "thread-existing"
        )
        assert "--sandbox" not in called_cmd
        assert "--output-last-message" not in called_cmd
        assert response.session_id == "thread-existing"
        assert response.content == "continued"

    async def test_execute_command_extracts_completed_response_text(
        self, manager: ClaudeSDKManager
    ):
        async def _create_process(*cmd, **kwargs):
            cmd_list = list(cmd)
            output_index = cmd_list.index("--output-last-message") + 1
            output_path = Path(cmd_list[output_index])
            output_path.write_text("", encoding="utf-8")

            payload = {
                "type": "response.completed",
                "response": {
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": "I can see repositories in this directory.",
                                }
                            ],
                        }
                    ]
                },
            }
            stdout_lines = [json.dumps(payload).encode("utf-8") + b"\n"]
            return _MockProcess(stdout_lines=stdout_lines, returncode=0)

        with patch(
            "src.claude.sdk_integration.asyncio.create_subprocess_exec",
            side_effect=_create_process,
        ):
            response = await manager.execute_command(
                prompt="can you see repos?",
                working_directory=Path("/tmp"),
            )

        assert response.content == "I can see repositories in this directory."

    async def test_execute_command_resume_strips_sandbox_extra_args(
        self, manager: ClaudeSDKManager
    ):
        manager.config.codex_extra_args = ["--sandbox", "workspace-write", "--search"]
        called_cmd = []

        async def _create_process(*cmd, **kwargs):
            cmd_list = list(cmd)
            called_cmd.extend(cmd_list)

            return _MockProcess(
                stdout_lines=[
                    b'{"type":"turn.started"}\n',
                    b'{"type":"response.output_text.delta","delta":"continued"}\n',
                ],
                returncode=0,
            )

        with patch(
            "src.claude.sdk_integration.asyncio.create_subprocess_exec",
            side_effect=_create_process,
        ):
            await manager.execute_command(
                prompt="continue",
                working_directory=Path("/tmp"),
                session_id="thread-existing",
                continue_session=True,
            )

        assert "--sandbox" not in called_cmd
        assert "--search" in called_cmd
        assert "--output-last-message" not in called_cmd

    async def test_execute_command_stream_callback(self, manager: ClaudeSDKManager):
        updates = []

        async def _stream_callback(update: StreamUpdate):
            updates.append(update)

        async def _create_process(*cmd, **kwargs):
            cmd_list = list(cmd)
            output_index = cmd_list.index("--output-last-message") + 1
            output_path = Path(cmd_list[output_index])
            output_path.write_text("done", encoding="utf-8")

            stdout_lines = [
                b'{"type":"thread.started","thread_id":"thread-abc"}\n',
                b'{"type":"response.output_text.delta","delta":"partial"}\n',
                b'{"type":"exec.command.started","command":"pytest -q"}\n',
            ]
            return _MockProcess(stdout_lines=stdout_lines, returncode=0)

        with patch(
            "src.claude.sdk_integration.asyncio.create_subprocess_exec",
            side_effect=_create_process,
        ):
            await manager.execute_command(
                prompt="run tests",
                working_directory=Path("/tmp"),
                stream_callback=_stream_callback,
            )

        assert any(update.content == "partial" for update in updates)
        assert any(update.tool_calls for update in updates)

    async def test_execute_command_not_logged_in_error(self, manager: ClaudeSDKManager):
        async def _create_process(*cmd, **kwargs):
            cmd_list = list(cmd)
            output_index = cmd_list.index("--output-last-message") + 1
            output_path = Path(cmd_list[output_index])
            output_path.write_text("", encoding="utf-8")

            return _MockProcess(
                stdout_lines=[],
                stderr_lines=[b"Not logged in\n"],
                returncode=1,
            )

        with patch(
            "src.claude.sdk_integration.asyncio.create_subprocess_exec",
            side_effect=_create_process,
        ):
            with pytest.raises(ClaudeProcessError) as exc_info:
                await manager.execute_command(
                    prompt="hello",
                    working_directory=Path("/tmp"),
                )

        assert "not logged in" in str(exc_info.value).lower()

    async def test_execute_command_no_last_message_warning_is_nonfatal(
        self, manager: ClaudeSDKManager
    ):
        async def _create_process(*cmd, **kwargs):
            cmd_list = list(cmd)
            output_index = cmd_list.index("--output-last-message") + 1
            output_path = Path(cmd_list[output_index])
            output_path.write_text("", encoding="utf-8")

            return _MockProcess(
                stdout_lines=[b'{"type":"response.output_text.delta","delta":"partial"}\n'],
                stderr_lines=[
                    b"Warning: no last agent message; wrote empty content to /tmp/out.txt\n"
                ],
                returncode=1,
            )

        with patch(
            "src.claude.sdk_integration.asyncio.create_subprocess_exec",
            side_effect=_create_process,
        ):
            response = await manager.execute_command(
                prompt="hello",
                working_directory=Path("/tmp"),
            )

        assert response.content == "partial"

    async def test_execute_command_timeout(self, manager: ClaudeSDKManager):
        # Make timeout short so test stays fast.
        manager.config.claude_timeout_seconds = 1

        async def _create_process(*cmd, **kwargs):
            cmd_list = list(cmd)
            output_index = cmd_list.index("--output-last-message") + 1
            output_path = Path(cmd_list[output_index])
            output_path.write_text("", encoding="utf-8")

            return _MockProcess(
                stdout_lines=[b'{"type":"turn.started"}\n'],
                stderr_lines=[],
                returncode=0,
                delay=5.0,
            )

        with patch(
            "src.claude.sdk_integration.asyncio.create_subprocess_exec",
            side_effect=_create_process,
        ):
            with pytest.raises(ClaudeTimeoutError):
                await manager.execute_command(
                    prompt="slow request",
                    working_directory=Path("/tmp"),
                )

    def test_get_active_process_count(self, manager: ClaudeSDKManager):
        assert manager.get_active_process_count() == 0
