import logging
import shlex
from dataclasses import dataclass
from typing import Dict, List, Any, Protocol

from .config_manager import ConfigManager
from .mcp_clients import BaseMCPClient, StdioMCPClient, HttpMCPClient

logger = logging.getLogger(__name__)

@dataclass
class Tool:
    id: str
    name: str
    description: str
    input_schema: dict
    server_name: str

class ToolRAGProtocol(Protocol):
    def write_tools(self, docs: List[dict]) -> None:
        ...
        
    def get_tools(self, query: str, k: int) -> List[dict]:
        ...

class MCPManager:
    """
    Provides lazy tool discovery, semantic retrieval via RAG, and deterministic execution.
    Features robust transport decoupling and multi-transport support via configuration.
    """
    def __init__(self, rag_manager: ToolRAGProtocol):
        self.rag = rag_manager
        self.tools: Dict[str, Tool] = {}
        self.clients: Dict[str, BaseMCPClient] = {}
        self._synced_servers: set[str] = set()
        self.config_manager = ConfigManager()

    def load_from_config(self, path: str) -> None:
        """Load server configurations from a JSON file and dynamically mount clients via transport factory."""
        config = self.config_manager.load(path)
        servers = config.get("servers", [])
        
        for s_cfg in servers:
            name = s_cfg.get("name")
            if not name:
                logger.warning("Skipping server configuration without a 'name'.")
                continue
                
            try:
                client = self._create_client(s_cfg)
                self.clients[name] = client
                logger.debug(f"Registered MCP server '{name}' with transport '{s_cfg.get('transport', 'stdio')}'.")
            except Exception as e:
                logger.error(f"Failed to create MCP client for '{name}': {e}")

    def _create_client(self, config: dict) -> BaseMCPClient:
        """Factory method to resolve the correct MCP client instance by transport."""
        name = config["name"]
        transport = config.get("transport", "stdio").lower()
        
        if transport == "stdio":
            command = config.get("command", "")
            args = config.get("args", [])
            # Fallback wrapper if args isn't explicitly provided but heavily nested in command
            if not args and " " in command:
                parts = shlex.split(command)
                command = parts[0]
                args = parts[1:]
                
            return StdioMCPClient(name=name, command=command, args=args)
            
        elif transport == "http":
            url = config.get("url")
            headers = config.get("headers", {})
            timeout = config.get("timeout", 30)
            
            if not url:
                raise ValueError("HTTP transport requires a 'url' attribute.")
                
            return HttpMCPClient(name=name, url=url, headers=headers, timeout=timeout)
            
        else:
            raise ValueError(f"Unknown MCP transport type '{transport}'. Supported types: stdio, http")

    def _sync_server_tools(self, server_name: str, force: bool = False) -> None:
        """Lazily sync tools from a specific server to the global registry and RAG index."""
        if server_name not in self.clients:
            logger.warning(f"Server '{server_name}' not found.")
            return
            
        if server_name in self._synced_servers and not force:
            return
            
        client = self.clients[server_name]
        try:
            raw_tools = client.list_tools()
            rag_docs = []
            
            for t in raw_tools:
                t_name = t.get("name", "")
                t_desc = t.get("description", "")
                t_schema = t.get("inputSchema", {})
                
                tool_id = f"{server_name}.{t_name}"
                
                tool_obj = Tool(
                    id=tool_id,
                    name=t_name,
                    description=t_desc,
                    input_schema=t_schema,
                    server_name=server_name
                )
                self.tools[tool_id] = tool_obj
                
                rag_docs.append({
                    "id": tool_id,
                    "text": t_desc,
                    "metadata": {
                        "server": server_name
                    }
                })
                
            if rag_docs:
                self.rag.write_tools(rag_docs)
                
            self._synced_servers.add(server_name)
            logger.info(f"Synced {len(raw_tools)} tools from server '{server_name}'")
            
        except Exception as e:
            logger.error(f"Error syncing tools for server '{server_name}': {e}")

    def get_tools(self, query: str, k: int = 3) -> List[Tool]:
        """Semantic retrieval of tools based on natural language descriptions."""
        for server_name in self.clients.keys():
            if server_name not in self._synced_servers:
                self._sync_server_tools(server_name)
                
        results = self.rag.get_tools(query, k)
        
        retrieved_tools = []
        for result in results:
            t_id = result.get("id")
            if t_id and t_id in self.tools:
                retrieved_tools.append(self.tools[t_id])
                
        return retrieved_tools

    def execute_tool(self, tool_id: str, input_args: dict) -> dict:
        """Deterministically resolves and executes a tool mapping via respective clients."""
        if tool_id not in self.tools:
            parts = tool_id.split(".")
            if len(parts) == 2:
                server_name = parts[0]
                if server_name in self.clients:
                    self._sync_server_tools(server_name)
                    
        tool = self.tools.get(tool_id)
        if not tool:
            return {"error": f"Tool '{tool_id}' not found locally or in any synced server registry."}
            
        client = self.clients.get(tool.server_name)
        if not client:
            return {"error": f"Server '{tool.server_name}' not found for tool '{tool_id}'."}
            
        try:
            result = client.call_tool(tool.name, input_args)
            return {"result": result}
        except Exception as e:
            logger.error(f"Error executing tool '{tool_id}': {e}")
            self._sync_server_tools(tool.server_name, force=True)
            return {"error": str(e)}

    def cleanup(self):
        """Cleanly closes up any active clients and transports securely."""
        for name, client in self.clients.items():
            try:
                client.close()
            except Exception as e:
                logger.error(f"Error closing client '{name}': {e}")
