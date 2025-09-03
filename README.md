# FastGTD MCP Tools

Standalone MCP (Model Context Protocol) server that provides AI assistants with tools to interact with FastGTD API.

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

### Production Setup

Copy `.env.sample` to `.env` and configure for your production environment:

```bash
cp .env.sample .env
```

Edit `.env` with your FastGTD account credentials:

```env
FASTGTD_API_URL=http://localhost:8003
FASTGTD_USERNAME=your-email@example.com
FASTGTD_PASSWORD=your-password
LOG_DIR=./mcplogs
LOG_RETENTION_DAYS=30
```

### Testing Setup

For running tests, copy `.env.test.sample` to `.env.test`:

```bash
cp .env.test.sample .env.test
```

Edit `.env.test` with a dedicated test user account:

```env
FASTGTD_API_URL=http://localhost:8003
FASTGTD_USERNAME=testuser@example.com
FASTGTD_PASSWORD=testpass
LOG_DIR=./test_mcplogs
LOG_RETENTION_DAYS=1
```

**Important**: Use a separate test user account to avoid interfering with your production data during testing.

### Authentication Methods

The MCP server supports two authentication methods:

**Option 1: Username/Password (Recommended)**
- Automatically authenticates using FastGTD's `/auth/login` endpoint
- Caches tokens for better performance
- Used by both production and test configurations

**Option 2: Direct Token**
```env
FASTGTD_TOKEN=your_jwt_token_here
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

## Testing

Run the comprehensive test suite to verify all MCP functions:

```bash
python test_all_mcp_functions.py
```

The test script automatically:
- Uses the `.env.test` configuration (separate from production)
- Tests all 22 MCP functions
- Creates and cleans up test data
- Reports detailed results

**Prerequisites for testing:**
1. Set up test environment: `cp .env.test.sample .env.test`
2. Create a test user account in FastGTD (separate from your main account)
3. Update `.env.test` with test user credentials

## Requirements

- Python 3.8+
- Access to a running FastGTD API server
- Valid JWT authentication token

## Environment Variables

### Production (.env)
- `FASTGTD_API_URL`: FastGTD API base URL (default: http://localhost:8003)
- `FASTGTD_USERNAME`: Your FastGTD account email
- `FASTGTD_PASSWORD`: Your FastGTD account password
- `FASTGTD_TOKEN`: Direct JWT token (alternative to username/password)
- `LOG_DIR`: Directory for MCP server logs (default: ./mcplogs)
- `LOG_LEVEL`: Logging level (default: INFO)
- `LOG_RETENTION_DAYS`: Days to keep logs before auto-cleanup (default: 30)

### Testing (.env.test)
- Same variables as production, but with test user credentials
- `LOG_DIR`: Separate test log directory (default: ./test_mcplogs)
- `LOG_RETENTION_DAYS`: Shorter retention for tests (default: 1)

### Log Management
Logs are automatically cleaned up on server startup based on `LOG_RETENTION_DAYS`:
- `0`: Delete all logs on each startup (no history)
- `1-365`: Keep logs for specified number of days
- Default: `30` days for production, `1` day for testing