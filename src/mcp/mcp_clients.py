import abc
import asyncio
import threading
import logging
from typing import List, Dict, Optional
from contextlib import AsyncExitStack

try:
    import httpx
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.sse import sse_client
except ImportError:
    ClientSession = None
    StdioServerParameters = None
    stdio_client = None
    sse_client = None

logger = logging.getLogger(__name__)

class BaseMCPClient(abc.ABC):
    """Abstract base class for all MCP clients across different transports."""
    def __init__(self, name: str):
        self.name = name

    @abc.abstractmethod
    def list_tools(self) -> List[dict]:
        """Fetch available tools from the server."""
        pass

    @abc.abstractmethod
    def call_tool(self, name: str, arguments: dict) -> dict:
        """Execute a tool on the server."""
        pass

    @abc.abstractmethod
    def close(self) -> None:
        """Gracefully close the client connection."""
        pass

class StdioMCPClient(BaseMCPClient):
    """MCP client using STDIO transport for local processes."""
    def __init__(self, name: str, command: str, args: List[str]):
        super().__init__(name)
        self.command = command
        self.args = args
        
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        
        self.session: Optional[ClientSession] = None
        self._exit_stack = AsyncExitStack()

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run_coro(self, coro):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    async def _initialize_async(self):
        if ClientSession is None:
            raise ImportError("mcp library is not installed.")
            
        server_params = StdioServerParameters(command=self.command, args=self.args, env=None)
        stdio_transport = await self._exit_stack.enter_async_context(stdio_client(server_params))
        read, write = stdio_transport
        self.session = await self._exit_stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()

    def list_tools(self) -> List[dict]:
        if self.session is None:
            self._run_coro(self._initialize_async())
        
        result = self._run_coro(self.session.list_tools())
        tools = []
        for t in result.tools:
            tools.append({
                "name": t.name,
                "description": getattr(t, 'description', ""),
                "inputSchema": getattr(t, 'inputSchema', {})
            })
        return tools

    def call_tool(self, name: str, arguments: dict) -> dict:
        if self.session is None:
            self._run_coro(self._initialize_async())
            
        result = self._run_coro(self.session.call_tool(name, arguments=arguments))
        content = []
        for c in result.content:
            if getattr(c, 'type', None) == "text":
                content.append({"text": getattr(c, 'text', '')})
            else:
                content.append({"type": getattr(c, 'type', 'unknown'), "data": str(c)})
        return {"content": content, "isError": getattr(result, 'isError', False)}
        
    def close(self):
        if self.session is not None:
            self._run_coro(self._exit_stack.aclose())
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()

class HttpMCPClient(BaseMCPClient):
    """MCP client using HTTP SSE transport for remote servers."""
    def __init__(self, name: str, url: str, headers: Dict[str, str], timeout: int = 30):
        super().__init__(name)
        self.url = url
        self.headers = headers
        self.timeout = timeout
        
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        
        self.session: Optional[ClientSession] = None
        self._exit_stack = AsyncExitStack()

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run_coro(self, coro):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    async def _initialize_async(self):
        if ClientSession is None:
            raise ImportError("mcp library is not installed.")
            
        kwargs = {}
        if self.timeout is not None:
            kwargs['timeout'] = self.timeout
            
        try:
            # httpx allows passing kwargs directly through to sse_client
            transport = await self._exit_stack.enter_async_context(
                sse_client(self.url, **kwargs)
            )
            read, write = transport
            self.session = await self._exit_stack.enter_async_context(ClientSession(read, write))
            await self.session.initialize()
        except Exception as e:
            logger.error(f"HTTP MCP Client '{self.name}' init failed: {e}")
            raise e

    def list_tools(self) -> List[dict]:
        if self.session is None:
            self._run_coro(self._initialize_async())
            
        result = self._run_coro(self.session.list_tools())
        tools = []
        for t in result.tools:
            tools.append({
                "name": t.name,
                "description": getattr(t, 'description', ""),
                "inputSchema": getattr(t, 'inputSchema', {})
            })
        return tools

    def call_tool(self, name: str, arguments: dict) -> dict:
        if self.session is None:
            self._run_coro(self._initialize_async())
            
        result = self._run_coro(self.session.call_tool(name, arguments=arguments))
        content = []
        for c in result.content:
            if getattr(c, 'type', None) == "text":
                content.append({"text": getattr(c, 'text', '')})
            else:
                content.append({"type": getattr(c, 'type', 'unknown'), "data": str(c)})
        return {"content": content, "isError": getattr(result, 'isError', False)}
        
    def close(self):
        if self.session is not None:
            self._run_coro(self._exit_stack.aclose())
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
