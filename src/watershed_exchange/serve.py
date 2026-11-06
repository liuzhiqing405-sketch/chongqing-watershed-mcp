"""A shared command-line front door for the two MCP servers.

Each server can be started two ways:

* ``stdio`` (the default) — a subprocess owned by whoever launched it. Simple,
  and the right choice for tests and one-off scripts, but the service and any
  job it runs end with that parent.
* ``streamable-http`` — a persistent loopback service. This is what the agent
  uses for real work: it connects, disconnects and reconnects without the
  service's jobs being affected.

The default host is ``127.0.0.1``. A research service holding local data has no
business being reachable from the local network, so binding elsewhere requires
saying so explicitly.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Callable, Sequence


DEFAULT_HOST = "127.0.0.1"


def build_parser(name: str, *, default_port: int,
                 version: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=name,
        description=f"启动 {name}（默认 stdio；长任务请用 streamable-http 常驻服务）",
    )
    parser.add_argument("--transport", choices=("stdio", "streamable-http"),
                        default="stdio", help="传输方式；默认 stdio")
    parser.add_argument("--host", default=DEFAULT_HOST,
                        help=f"HTTP 监听地址；默认 {DEFAULT_HOST}（仅本机）")
    parser.add_argument("--port", type=int, default=default_port,
                        help=f"HTTP 监听端口；默认 {default_port}")
    parser.add_argument("--endpoint", default="/mcp", help="HTTP 路径；默认 /mcp")
    parser.add_argument("--version", action="version",
                        version=f"{name} {version or 'unknown'}",
                        help="显示服务版本并退出")
    return parser


def serve(mcp: Any, name: str, *, default_port: int,
          before_serve: Callable[[], None] | None = None,
          argv: Sequence[str] | None = None) -> int:
    """Run ``mcp`` according to the command line."""
    # The version the server reports to its clients is the one it prints here,
    # so a startup log and a handshake can never disagree.
    version = str(getattr(mcp, "version", "") or "unknown")
    args = build_parser(name, default_port=default_port,
                        version=version).parse_args(argv)
    if args.transport == "stdio":
        mcp.run()
        return 0
    if not (1 <= args.port <= 65535):
        print(f"端口无效：{args.port}", file=sys.stderr)
        return 2
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        print(f"警告：正在监听 {args.host}，服务将可被本机之外访问。", file=sys.stderr)
    if before_serve is not None:
        before_serve()
    print(f"{name} {version} 正在监听 http://{args.host}:{args.port}{args.endpoint}",
          flush=True)
    mcp.run(transport="streamable-http", host=args.host, port=args.port,
            streamable_http_path=args.endpoint)
    return 0
