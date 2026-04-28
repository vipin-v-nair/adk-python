# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Thread-isolated MCP helpers for environments with anyio cancel scope constraints.

Root cause
----------
``anyio``'s ``CancelScope`` binds to the ``asyncio.Task`` that *enters* it.
On Vertex AI Agent Engine, the scheduler can context-switch tasks between
entering and exiting the scope inside ``streamablehttp_client``'s
``anyio.create_task_group()``, which raises::

    Attempted to exit cancel scope in a different task than it was entered in

Fix
---
Run each MCP operation inside a dedicated thread via ``asyncio.to_thread()``.
Inside that thread, ``asyncio.new_event_loop()`` creates an isolated event
loop. The ``anyio`` cancel scope is created *and* destroyed entirely within
that loop, so it never crosses task boundaries in the caller's scheduler.

Trade-offs
----------
* A new HTTP connection is opened per tool call (no session reuse).
* ``progress_callback`` and MCP sampling are not supported in this path.
* Auth headers are threaded through, so ``auth_scheme``/``auth_credential``
  and ``header_provider`` on ``McpToolset`` remain functional.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional

from mcp import types
from mcp.client import streamable_http
from mcp.client.session import ClientSession


def _cancel_pending(loop: asyncio.AbstractEventLoop) -> None:
  """Cancel all pending tasks so the loop can be closed without warnings."""
  pending = asyncio.all_tasks(loop)
  if not pending:
    return
  for task in pending:
    task.cancel()
  loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))


def list_tools_in_thread(
    url: str,
    headers: Optional[Dict[str, str]] = None,
) -> list[Any]:
  """Return tools/list results from an MCP server via an isolated event loop.

  Opens a fresh connection for every call; must be invoked from a non-async
  context (i.e. via ``asyncio.to_thread``).

  Args:
    url: The MCP server URL.
    headers: Optional HTTP headers to include in the request.

  Returns:
    A list of ``mcp.types.Tool`` objects.
  """

  async def _async():
    kwargs = {"headers": headers} if headers else {}
    async with streamable_http.streamable_http_client(url, **kwargs) as (
        read,
        write,
        _,
    ):
      async with ClientSession(read, write) as session:
        await session.initialize()
        result = await session.list_tools()
        return result.tools

  loop = asyncio.new_event_loop()
  asyncio.set_event_loop(loop)
  try:
    return loop.run_until_complete(_async())
  finally:
    _cancel_pending(loop)
    loop.close()


def call_tool_in_thread(
    url: str,
    tool_name: str,
    arguments: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
) -> Any:
  """Call an MCP tool via an isolated event loop and return the parsed result.

  Opens a fresh connection for every call; must be invoked from a non-async
  context (i.e. via ``asyncio.to_thread``).

  Args:
    url: The MCP server URL.
    tool_name: The name of the tool to call.
    arguments: The arguments to pass to the tool.
    headers: Optional HTTP headers to include in the request.

  Returns:
    The parsed tool result (JSON-decoded dict, or ``{"text": ...}`` fallback).

  Raises:
    RuntimeError: If the MCP server returns an error response.
  """

  async def _async():
    kwargs = {"headers": headers} if headers else {}
    async with streamable_http.streamable_http_client(url, **kwargs) as (
        read,
        write,
        _,
    ):
      async with ClientSession(read, write) as session:
        await session.initialize()
        result = await session.call_tool(tool_name, arguments)
        if result.isError:
          raise RuntimeError(
              f"MCP tool '{tool_name}' returned an error: {result.content}"
          )
        if result.content:
          first = result.content[0]
          if isinstance(first, types.TextContent):
            try:
              return json.loads(first.text)
            except json.JSONDecodeError:
              return {"text": first.text}
          return {"content": str(first)}
        return {}


  loop = asyncio.new_event_loop()
  asyncio.set_event_loop(loop)
  try:
    return loop.run_until_complete(_async())
  finally:
    _cancel_pending(loop)
    loop.close()
