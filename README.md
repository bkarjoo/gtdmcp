# FastGTD MCP Tools

Standalone MCP (Model Context Protocol) server that provides AI assistants with tools to interact with FastGTD API.

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Copy `.env.sample` to `.env` and configure:

```bash
cp .env.sample .env
```

Edit `.env`:
```
FASTGTD_API_URL=http://localhost:8003
LOG_DIR=./mcplogs
```

## Usage

### As MCP Server (stdio)
```bash
python fastgtd_mcp_server.py
```

### In MCP Client Configuration

Add to your MCP client config:
```json
{
  "mcpServers": {
    "fastgtd": {
      "command": "python",
      "args": ["/path/to/gtdmcp/fastgtd_mcp_server.py"],
      "sendAuth": true,
      "sendCurrentNode": true
    }
  }
}
```

## Available Tools

- **Task Management**: Create, update, complete, delete tasks
- **Note Management**: Create and update notes
- **Folder Management**: Create folders and manage hierarchy
- **Search & Navigation**: Search nodes, browse tree structure
- **Smart Folders**: Get smart folder contents
- **Templates**: List and instantiate templates
- **Tags**: Add and remove tags from nodes

## Requirements

- Python 3.8+
- Access to a running FastGTD API server
- Valid JWT authentication token

## Environment Variables

- `FASTGTD_API_URL`: FastGTD API base URL (default: http://localhost:8003)
- `LOG_DIR`: Directory for MCP server logs (default: ./mcplogs)
- `LOG_LEVEL`: Logging level (default: INFO)