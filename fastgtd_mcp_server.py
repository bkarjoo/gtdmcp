#!/usr/bin/env python3
"""
FastGTD MCP Server - Test Implementation
Simple MCP server with one test tool to verify authentication headers.
"""

import asyncio
import json
import logging
from datetime import datetime
import os
import time
from pathlib import Path
import httpx
from dotenv import load_dotenv
from mcp.server import Server

# Load environment variables
load_dotenv()

# Configuration from environment variables
FASTGTD_API_URL = os.getenv('FASTGTD_API_URL', 'http://localhost:8003')
LOG_DIR = os.getenv('LOG_DIR', '/tmp/fastgtd_mcp_logs')
LOG_RETENTION_DAYS = int(os.getenv('LOG_RETENTION_DAYS', '30'))

# Authentication configuration
FASTGTD_TOKEN = os.getenv('FASTGTD_TOKEN')
FASTGTD_USERNAME = os.getenv('FASTGTD_USERNAME') 
FASTGTD_PASSWORD = os.getenv('FASTGTD_PASSWORD')

# File download configuration
DEFAULT_DOWNLOAD_PATH = os.getenv('DEFAULT_DOWNLOAD_PATH', '/tmp')

# Pagination constants
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 100
MAX_SEARCH_RESULTS = 100

# Response limits
CHARACTER_LIMIT = 25000  # Max characters in response (MCP best practice)
MAX_TREE_DEPTH = 20
DEFAULT_TREE_DEPTH = 10

# HTTP configuration
HTTP_TIMEOUT = 30.0  # seconds
MAX_RETRIES = 3

# Response format configuration
DEFAULT_RESPONSE_FORMAT = "markdown"  # "json" or "markdown"
ALLOWED_FORMATS = ["json", "markdown"]
DEFAULT_DETAIL_LEVEL = "concise"  # "concise" or "detailed"
ALLOWED_DETAIL_LEVELS = ["concise", "detailed"]

from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    TextContent,
    Tool,
)

# Pydantic imports for input validation
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Literal, Optional
from enum import Enum

# Enum types for constrained values
class Priority(str, Enum):
    """Task priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class NodeType(str, Enum):
    """Node types in FastGTD"""
    TASK = "task"
    NOTE = "note"
    FOLDER = "folder"
    SMART_FOLDER = "smart_folder"
    TEMPLATE = "template"

class TaskStatus(str, Enum):
    """Task completion status"""
    INCOMPLETE = "incomplete"
    COMPLETE = "complete"

# Pydantic models for input validation
class CreateTaskInput(BaseModel):
    """Input model for create_task"""
    model_config = ConfigDict(extra="forbid")

    title: str = Field(..., min_length=1, max_length=500, description="Task title")
    description: str = Field("", max_length=5000, description="Task description")
    priority: Optional[Priority] = Field(None, description="Task priority level (low, medium, high, urgent)")
    due_at: Optional[str] = Field(None, description="Due date/time in ISO format (e.g., '2024-12-25T10:00:00Z')")
    parent_id: Optional[str] = Field(None, description="Parent folder ID")
    tags: Optional[list[str]] = Field(None, description="List of tags to add")
    recurrence_rule: Optional[str] = Field(None, description="Recurrence rule in iCalendar format")

class SearchNodesInput(BaseModel):
    """Input model for search_nodes"""
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, description="Search query text")
    limit: int = Field(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE, description=f"Max results (1-{MAX_PAGE_SIZE})")
    node_types: Optional[list[NodeType]] = Field(None, description="Filter by node types")

class PaginationInput(BaseModel):
    """Standard pagination parameters"""
    model_config = ConfigDict(extra="forbid")

    limit: int = Field(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE, description=f"Results per page (default: {DEFAULT_PAGE_SIZE}, max: {MAX_PAGE_SIZE})")
    offset: int = Field(0, ge=0, description="Number of items to skip (default: 0)")

class ResponseFormatInput(BaseModel):
    """Response format configuration"""
    model_config = ConfigDict(extra="forbid")

    response_format: Literal["json", "markdown"] = Field(DEFAULT_RESPONSE_FORMAT, description="Response format (json or markdown)")
    detail_level: Literal["concise", "detailed"] = Field(DEFAULT_DETAIL_LEVEL, description="Detail level (concise or detailed)")

class UpdateTaskInput(BaseModel):
    """Input model for update_task"""
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(..., min_length=1, description="Task ID to update")
    title: Optional[str] = Field(None, min_length=1, max_length=500, description="Task title")
    description: Optional[str] = Field(None, max_length=5000, description="Task description")
    priority: Optional[Priority] = Field(None, description="Task priority level")
    status: Optional[TaskStatus] = Field(None, description="Task completion status")
    due_at: Optional[str] = Field(None, description="Due date/time in ISO format")
    earliest_start_at: Optional[str] = Field(None, description="Earliest start date/time in ISO format")
    archived: Optional[bool] = Field(None, description="Archive status")
    recurrence_rule: Optional[str] = Field(None, description="Recurrence rule in iCalendar format")
    recurrence_anchor: Optional[str] = Field(None, description="Recurrence anchor date")

class CreateFolderInput(BaseModel):
    """Input model for create_folder"""
    model_config = ConfigDict(extra="forbid")

    title: str = Field(..., min_length=1, max_length=500, description="Folder title")
    description: Optional[str] = Field("", max_length=5000, description="Folder description")
    parent_id: Optional[str] = Field(None, description="Parent folder ID")

class SearchTemplatesInput(BaseModel):
    """Input model for search_templates"""
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, description="Search query text")
    category: Optional[str] = Field("", description="Filter by category")
    limit: int = Field(50, ge=1, le=MAX_PAGE_SIZE, description=f"Max results (1-{MAX_PAGE_SIZE})")
    offset: int = Field(0, ge=0, description="Offset for pagination")

# Utility functions for response formatting
def truncate_response(text: str, limit: int = CHARACTER_LIMIT) -> str:
    """Truncate response if it exceeds character limit"""
    if len(text) <= limit:
        return text

    truncated = text[:limit - 100]  # Leave room for truncation message
    return f"{truncated}\n\n... [Response truncated. Original length: {len(text)} characters, showing first {limit - 100} characters]"

def format_response_as_markdown(data: dict, detail_level: str = "concise") -> str:
    """Format response data as human-readable Markdown"""
    if not data.get("success", False):
        # Error responses stay as JSON for clarity
        return json.dumps(data, indent=2)

    lines = []

    # Handle different response types
    if "nodes" in data or "results" in data:
        # List of nodes/results
        items = data.get("nodes") or data.get("results", [])
        if detail_level == "concise":
            lines.append(f"**Found {len(items)} items:**\n")
            for item in items:
                title = item.get("title") or item.get("name", "Untitled")
                node_type = item.get("node_type", "item")
                node_id = item.get("id", "")
                lines.append(f"- [{node_type}] {title} (ID: {node_id})")
        else:  # detailed
            lines.append(f"**Found {len(items)} items:**\n")
            for item in items:
                title = item.get("title") or item.get("name", "Untitled")
                node_type = item.get("node_type", "item")
                node_id = item.get("id", "")
                lines.append(f"### {title}")
                lines.append(f"- **Type:** {node_type}")
                lines.append(f"- **ID:** {node_id}")
                if item.get("description"):
                    lines.append(f"- **Description:** {item['description']}")
                if item.get("priority"):
                    lines.append(f"- **Priority:** {item['priority']}")
                if item.get("due_at"):
                    lines.append(f"- **Due:** {item['due_at']}")
                lines.append("")

    elif "tree" in data:
        # Tree structure
        tree = data["tree"]
        lines.append(f"**Tree from '{tree.get('title', 'Root')}':**\n")
        lines.append(_format_tree_node(tree, level=0, detail_level=detail_level))

    elif "task" in data or "folder" in data or "note" in data:
        # Single item
        item = data.get("task") or data.get("folder") or data.get("note", {})
        title = item.get("title") or item.get("name", "Untitled")
        lines.append(f"## {title}")
        lines.append(f"- **ID:** {item.get('id', 'N/A')}")
        lines.append(f"- **Type:** {item.get('node_type', 'N/A')}")
        if detail_level == "detailed":
            if item.get("description"):
                lines.append(f"- **Description:** {item['description']}")
            if item.get("priority"):
                lines.append(f"- **Priority:** {item['priority']}")
            if item.get("due_at"):
                lines.append(f"- **Due:** {item['due_at']}")

    elif "message" in data:
        # Simple success message
        lines.append(f"**Success:** {data['message']}")
        if detail_level == "detailed" and len(data) > 2:
            lines.append(f"\n```json\n{json.dumps(data, indent=2)}\n```")

    else:
        # Fallback to JSON
        return json.dumps(data, indent=2)

    return "\n".join(lines)

def _format_tree_node(node: dict, level: int = 0, detail_level: str = "concise") -> str:
    """Recursively format a tree node"""
    indent = "  " * level
    title = node.get("title") or node.get("name", "Untitled")
    node_type = node.get("node_type", "item")
    node_id = node.get("id", "")

    lines = []
    if detail_level == "concise":
        lines.append(f"{indent}- [{node_type}] {title}")
    else:
        lines.append(f"{indent}- [{node_type}] {title} (ID: {node_id})")
        if node.get("description"):
            lines.append(f"{indent}  Description: {node['description'][:100]}...")

    # Recursively format children
    if node.get("children"):
        for child in node["children"]:
            lines.append(_format_tree_node(child, level + 1, detail_level))

    return "\n".join(lines)

def format_response(
    data: dict,
    response_format: str = DEFAULT_RESPONSE_FORMAT,
    detail_level: str = DEFAULT_DETAIL_LEVEL
) -> str:
    """Format response based on format and detail level preferences"""
    if response_format == "markdown":
        formatted = format_response_as_markdown(data, detail_level)
    else:  # json
        formatted = json.dumps(data, indent=2)

    # Always truncate if needed
    return truncate_response(formatted)

# Error handling utilities
def create_error_response(
    error_message: str,
    suggestion: str = "",
    error_code: str = "",
    details: dict = None
) -> dict:
    """
    Create a standardized, actionable error response.

    Args:
        error_message: Clear description of what went wrong
        suggestion: Actionable suggestion for how to fix it
        error_code: Machine-readable error code (e.g., "AUTH_FAILED", "MISSING_PARAM")
        details: Additional context/details

    Returns:
        Standardized error response dict
    """
    response = {
        "success": False,
        "error": error_message
    }

    if suggestion:
        response["error"] = f"{error_message}. {suggestion}"

    if error_code:
        response["error_code"] = error_code

    if details:
        response["details"] = details

    return response

# Common error templates
ERROR_TEMPLATES = {
    "no_auth": {
        "message": "No authentication token available",
        "suggestion": "Ensure FASTGTD_TOKEN environment variable is set, or FASTGTD_USERNAME and FASTGTD_PASSWORD are configured",
        "code": "AUTH_MISSING"
    },
    "node_id_required": {
        "message": "Node ID is required",
        "suggestion": "Use get_folder_id(folder_name='YourFolder') to find a folder ID, or get_root_nodes() to list available nodes",
        "code": "MISSING_NODE_ID"
    },
    "search_query_required": {
        "message": "Search query is required and must be at least 1 character",
        "suggestion": "Provide a search term, e.g., search_nodes(query='meeting') to find items containing 'meeting'",
        "code": "INVALID_QUERY"
    },
    "invalid_node_type": {
        "message": "Invalid node type",
        "suggestion": "Valid node types are: 'task', 'note', 'folder', 'smart_folder', 'template'",
        "code": "INVALID_NODE_TYPE"
    },
    "invalid_priority": {
        "message": "Invalid priority",
        "suggestion": "Valid priorities are: 'low', 'medium', 'high', 'urgent'",
        "code": "INVALID_PRIORITY"
    }
}

def get_error_response(template_key: str, **kwargs) -> dict:
    """Get a standardized error response from a template."""
    template = ERROR_TEMPLATES.get(template_key, {})
    return create_error_response(
        error_message=template.get("message", "An error occurred"),
        suggestion=template.get("suggestion", ""),
        error_code=template.get("code", ""),
        details=kwargs
    )

def cleanup_old_logs(log_dir, days_old=30):
    """Remove log files older than specified days. If days_old=0, delete all logs."""
    try:
        log_path = Path(log_dir)
        if not log_path.exists():
            return
        
        deleted_count = 0
        
        if days_old == 0:
            # Delete all log files
            for log_file in log_path.glob("*.log"):
                log_file.unlink()
                deleted_count += 1
            if deleted_count > 0:
                print(f"🧹 Cleared all {deleted_count} log files (LOG_RETENTION_DAYS=0)")
        else:
            # Delete files older than specified days
            cutoff_time = time.time() - (days_old * 24 * 60 * 60)
            
            for log_file in log_path.glob("*.log"):
                if log_file.stat().st_mtime < cutoff_time:
                    log_file.unlink()
                    deleted_count += 1
            
            if deleted_count > 0:
                print(f"🧹 Cleaned up {deleted_count} old log files (older than {days_old} days)")
    except Exception as e:
        print(f"⚠️ Log cleanup failed: {e}")

# Set up file logging
os.makedirs(LOG_DIR, exist_ok=True)

# Clean up old logs on startup
cleanup_old_logs(LOG_DIR, days_old=LOG_RETENTION_DAYS)

log_file = os.path.join(LOG_DIR, f"mcp_server_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Configure logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info(f"=== FastGTD MCP Server Starting - Log file: {log_file} ===")

# Initialize MCP server
# Global authentication token cache
_cached_token = None

async def get_auth_token() -> str:
    """Get authentication token - either from env var or by logging in"""
    global _cached_token
    
    # If we have a direct token from env, use it
    if FASTGTD_TOKEN:
        return FASTGTD_TOKEN
    
    # If we have cached token, use it
    if _cached_token:
        return _cached_token
        
    # If we have username/password, login to get token
    if FASTGTD_USERNAME and FASTGTD_PASSWORD:
        import httpx
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{FASTGTD_API_URL}/auth/login",
                    json={
                        "email": FASTGTD_USERNAME,
                        "password": FASTGTD_PASSWORD
                    },
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    _cached_token = data["access_token"]
                    logger.info("🔐 Successfully authenticated with FastGTD API")
                    return _cached_token
                else:
                    logger.error(f"❌ Authentication failed: {response.status_code} - {response.text}")
                    return ""
        except Exception as e:
            logger.error(f"❌ Authentication error: {e}")
            return ""
    
    # No authentication configured
    logger.warning("⚠️  No authentication configured - operations will use passed token")
    return ""

server = Server("fastgtd-mcp")

async def add_task_to_inbox(title: str, description: str = "", priority: str = "medium", auth_token: str = "", current_node_id: str = "") -> dict:
    """Add a task to the user's default node (inbox)"""
    logger.info(f"add_task_to_inbox called - title='{title}', priority='{priority}'")

    # Validate inputs
    if not title or len(title.strip()) == 0:
        return get_error_response("node_id_required")  # Reusing this as it's similar

    # Validate priority
    valid_priorities = ["low", "medium", "high", "urgent"]
    if priority and priority not in valid_priorities:
        return get_error_response("invalid_priority")

    # Get auth token if not provided
    if not auth_token:
        auth_token = await get_auth_token()

    if not auth_token:
        return get_error_response("no_auth")

    # FastGTD API endpoint
    url = f"{FASTGTD_API_URL}/nodes/"

    # Create task payload for unified node system
    task_payload = {
        "node_type": "task",
        "title": title.strip(),
        "task_data": {
            "description": description,
            "priority": priority,
            "status": "todo",
            "archived": False
        }
    }

    # Prepare headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_token}"
    }

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            # First, get the user's default node (inbox)
            default_node_response = await client.get(
                f"{FASTGTD_API_URL}/settings/default-node",
                headers=headers
            )

            if default_node_response.status_code == 200:
                default_data = default_node_response.json()
                default_node_id = default_data.get("node_id")
                if default_node_id:
                    task_payload["parent_id"] = default_node_id
                    logger.debug(f"Using default node: {default_node_id}")
                else:
                    logger.debug("No default node set - adding to root")
            else:
                logger.warning(f"Failed to get default node: HTTP {default_node_response.status_code}")

            # Create the task
            response = await client.post(
                url,
                json=task_payload,
                headers=headers,
                timeout=HTTP_TIMEOUT
            )

            if response.status_code in [200, 201]:
                task_data = response.json()
                return {
                    "success": True,
                    "message": f"Task '{title}' added to inbox successfully",
                    "task_id": task_data.get("id"),
                    "task": task_data
                }
            else:
                return create_error_response(
                    f"Failed to add task: HTTP {response.status_code}",
                    f"Check your authentication and try again",
                    "API_ERROR",
                    {"http_status": response.status_code, "response": response.text}
                )

    except httpx.TimeoutException:
        return create_error_response(
            "Request timed out",
            "The FastGTD API is not responding. Check your FASTGTD_API_URL and network connection",
            "TIMEOUT"
        )
    except Exception as e:
        logger.error(f"Failed to add task to inbox: {str(e)}")
        return create_error_response(
            f"Failed to add task to inbox: {str(e)}",
            "Check your network connection and API configuration",
            "EXCEPTION"
        )

async def add_folder_to_current_node(title: str, description: str = "", auth_token: str = "", current_node_id: str = "") -> dict:
    """Add a folder to the current node"""
    logger.info(f"add_folder_to_current_node called - title='{title}'")

    # Validate inputs
    if not title or len(title.strip()) == 0:
        return create_error_response("Title is required", "Provide a folder title", "MISSING_TITLE")

    if not current_node_id:
        return create_error_response(
            "No current node ID provided",
            "Use create_folder(parent_id='...') to specify a parent node",
            "MISSING_NODE_ID"
        )

    # Get auth token if not provided
    if not auth_token:
        auth_token = await get_auth_token()

    if not auth_token:
        return get_error_response("no_auth")

    # FastGTD API endpoint
    url = f"{FASTGTD_API_URL}/nodes/"

    # Create folder payload - folders are their own node type
    folder_payload = {
        "node_type": "folder",
        "title": title.strip(),
        "parent_id": current_node_id
    }

    # Add description if provided
    if description:
        folder_payload["folder_data"] = {"description": description}

    # Prepare headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_token}"
    }

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.post(
                url,
                json=folder_payload,
                headers=headers,
                timeout=HTTP_TIMEOUT
            )

            if response.status_code in [200, 201]:
                folder_data = response.json()
                return {
                    "success": True,
                    "message": f"Folder '{title}' added to current node successfully",
                    "folder_id": folder_data.get("id"),
                    "folder": folder_data
                }
            else:
                return create_error_response(
                    f"Failed to add folder: HTTP {response.status_code}",
                    "Check your authentication and parameters",
                    "API_ERROR",
                    {"http_status": response.status_code, "response": response.text}
                )

    except httpx.TimeoutException:
        return create_error_response(
            "Request timed out",
            "The FastGTD API is not responding. Check your FASTGTD_API_URL and network connection",
            "TIMEOUT"
        )
    except Exception as e:
        logger.error(f"Failed to add folder to current node: {str(e)}")
        return create_error_response(
            f"Failed to add folder to current node: {str(e)}",
            "Check your network connection and API configuration",
            "EXCEPTION"
        )

async def add_task_to_current_node(title: str, description: str = "", priority: str = "medium", auth_token: str = "", current_node_id: str = "") -> dict:
    """Add a task to the user's currently selected node"""
    logger.info(f"add_task_to_current_node called - title='{title}', priority='{priority}'")

    # Validate inputs
    if not title or len(title.strip()) == 0:
        return create_error_response("Title is required", "Provide a task title", "MISSING_TITLE")

    if not current_node_id:
        return create_error_response(
            "No current node ID provided",
            "Use add_task_to_inbox() or add_task_to_node_id(node_id='...') instead",
            "MISSING_NODE_ID"
        )

    # Validate priority
    valid_priorities = ["low", "medium", "high", "urgent"]
    if priority and priority not in valid_priorities:
        return get_error_response("invalid_priority")

    # Get auth token if not provided
    if not auth_token:
        auth_token = await get_auth_token()

    if not auth_token:
        return get_error_response("no_auth")

    # FastGTD API endpoint
    url = f"{FASTGTD_API_URL}/nodes/"

    # Create task payload for unified node system
    task_payload = {
        "node_type": "task",
        "title": title.strip(),
        "parent_id": current_node_id,
        "task_data": {
            "description": description,
            "priority": priority,
            "status": "todo",
            "archived": False
        }
    }

    # Prepare headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_token}"
    }

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.post(
                url,
                json=task_payload,
                headers=headers,
                timeout=HTTP_TIMEOUT
            )

            if response.status_code in [200, 201]:
                task_data = response.json()
                return {
                    "success": True,
                    "message": f"Task '{title}' added to current node successfully",
                    "task_id": task_data.get("id"),
                    "task": task_data
                }
            else:
                return create_error_response(
                    f"Failed to add task: HTTP {response.status_code}",
                    "Check your authentication and node ID",
                    "API_ERROR",
                    {"http_status": response.status_code, "response": response.text}
                )

    except httpx.TimeoutException:
        return create_error_response(
            "Request timed out",
            "The FastGTD API is not responding. Check your FASTGTD_API_URL and network connection",
            "TIMEOUT"
        )
    except Exception as e:
        logger.error(f"Failed to add task to current node: {str(e)}")
        return create_error_response(
            f"Failed to add task to current node: {str(e)}",
            "Check your network connection and API configuration",
            "EXCEPTION"
        )

async def add_note_to_current_node(title: str, content: str = "", auth_token: str = "", current_node_id: str = "") -> dict:
    """Add a note to the user's currently selected node"""
    logger.info(f"add_note_to_current_node called - title='{title}'")

    # Validate inputs
    if not title or len(title.strip()) == 0:
        return create_error_response("Title is required", "Provide a note title", "MISSING_TITLE")

    if not current_node_id:
        return create_error_response(
            "No current node ID provided",
            "Use add_note_to_node_id(node_id='...') to specify a parent node",
            "MISSING_NODE_ID"
        )

    # Get auth token if not provided
    if not auth_token:
        auth_token = await get_auth_token()

    if not auth_token:
        return get_error_response("no_auth")

    # FastGTD API endpoint
    url = f"{FASTGTD_API_URL}/nodes/"

    # Create note payload for unified node system
    note_payload = {
        "node_type": "note",
        "title": title.strip(),
        "parent_id": current_node_id,
        "note_data": {
            "body": content
        }
    }

    # Prepare headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_token}"
    }

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.post(
                url,
                json=note_payload,
                headers=headers,
                timeout=HTTP_TIMEOUT
            )

            if response.status_code in [200, 201]:
                note_data = response.json()
                return {
                    "success": True,
                    "message": f"Note '{title}' added to current node successfully",
                    "note_id": note_data.get("id"),
                    "note": note_data
                }
            else:
                return create_error_response(
                    f"Failed to add note: HTTP {response.status_code}",
                    "Check your authentication and parameters",
                    "API_ERROR",
                    {"http_status": response.status_code, "response": response.text}
                )

    except httpx.TimeoutException:
        return create_error_response(
            "Request timed out",
            "The FastGTD API is not responding. Check your FASTGTD_API_URL and network connection",
            "TIMEOUT"
        )
    except Exception as e:
        logger.error(f"Failed to add note to current node: {str(e)}")
        return create_error_response(
            f"Failed to add note to current node: {str(e)}",
            "Check your network connection and API configuration",
            "EXCEPTION"
        )

async def add_note_to_node_id(node_id: str, title: str, content: str = "", auth_token: str = "", current_node_id: str = "") -> dict:
    """Add a note to a specific node by its ID"""
    logger.info(f"add_note_to_node_id called - node_id='{node_id}', title='{title}'")

    # Validate inputs
    if not node_id or len(node_id.strip()) == 0:
        return get_error_response("node_id_required")

    if not title or len(title.strip()) == 0:
        return create_error_response("Title is required", "Provide a note title", "MISSING_TITLE")

    # Get auth token if not provided
    if not auth_token:
        auth_token = await get_auth_token()

    if not auth_token:
        return get_error_response("no_auth")

    # FastGTD API endpoint
    url = f"{FASTGTD_API_URL}/nodes/"

    # Create note payload for unified node system
    note_payload = {
        "node_type": "note",
        "title": title.strip(),
        "parent_id": node_id.strip(),
        "note_data": {
            "body": content
        }
    }

    # Prepare headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_token}"
    }

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.post(
                url,
                json=note_payload,
                headers=headers,
                timeout=HTTP_TIMEOUT
            )

            if response.status_code in [200, 201]:
                note_data = response.json()
                return {
                    "success": True,
                    "message": f"Successfully added note '{title}' to node {node_id}",
                    "note_id": note_data.get("id"),
                    "note": note_data
                }
            else:
                return create_error_response(
                    f"Failed to add note: HTTP {response.status_code}",
                    "Check your authentication and node ID",
                    "API_ERROR",
                    {"http_status": response.status_code, "response": response.text}
                )

    except httpx.TimeoutException:
        return create_error_response(
            "Request timed out",
            "The FastGTD API is not responding. Check your FASTGTD_API_URL and network connection",
            "TIMEOUT"
        )
    except Exception as e:
        logger.error(f"Failed to add note to node {node_id}: {str(e)}")
        return create_error_response(
            f"Failed to add note to node {node_id}: {str(e)}",
            "Check your network connection and API configuration",
            "EXCEPTION"
        )

async def get_all_folders(limit: int = DEFAULT_PAGE_SIZE, offset: int = 0, auth_token: str = "", current_node_id: str = "") -> dict:
    """Get all folder names in the user's node tree for AI to help find the right folder"""
    logger.info(f"get_all_folders called - limit={limit}, offset={offset}")

    # Get auth token if not provided
    if not auth_token:
        auth_token = await get_auth_token()

    if not auth_token:
        return get_error_response("no_auth")

    # FastGTD API endpoint
    url = f"{FASTGTD_API_URL}/nodes/"

    # Prepare headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_token}"
    }

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.get(
                url,
                headers=headers,
                params={
                    "node_type": "folder",
                    "limit": min(limit, MAX_PAGE_SIZE),
                    "offset": max(offset, 0)
                },
                timeout=HTTP_TIMEOUT
            )

            if response.status_code in [200, 201]:
                nodes_data = response.json()
                folders = []

                # Handle case where response might be None or not a list
                if nodes_data is None:
                    nodes_data = []
                elif not isinstance(nodes_data, list):
                    nodes_data = [nodes_data]

                # Extract folder data including descriptions
                for node in nodes_data:
                    if isinstance(node, dict) and node.get("node_type") == "folder":
                        folder_info = {
                            "id": node.get("id"),
                            "title": node.get("title"),
                            "description": node.get("folder_data", {}).get("description", "") if isinstance(node.get("folder_data"), dict) else ""
                        }
                        folders.append(folder_info)

                return {
                    "success": True,
                    "message": f"Found {len(folders)} folders",
                    "folders": folders
                }
            else:
                return create_error_response(
                    f"Failed to retrieve folders: HTTP {response.status_code}",
                    "Check your authentication",
                    "API_ERROR",
                    {"http_status": response.status_code, "response": response.text}
                )

    except httpx.TimeoutException:
        return create_error_response(
            "Request timed out",
            "The FastGTD API is not responding. Check your FASTGTD_API_URL and network connection",
            "TIMEOUT"
        )
    except Exception as e:
        logger.error(f"Failed to get folders: {str(e)}")
        return create_error_response(
            f"Failed to get folders: {str(e)}",
            "Check your network connection and API configuration",
            "EXCEPTION"
        )

async def get_root_folders(limit: int = DEFAULT_PAGE_SIZE, offset: int = 0, auth_token: str = "", current_node_id: str = "") -> dict:
    """Get only root-level folders (folders with no parent)"""
    logger.info(f"get_root_folders called - limit={limit}, offset={offset}")

    # Get auth token if not provided
    if not auth_token:
        auth_token = await get_auth_token()

    if not auth_token:
        return get_error_response("no_auth")

    # FastGTD API endpoint
    url = f"{FASTGTD_API_URL}/nodes/"

    # Prepare headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_token}"
    }

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.get(
                url,
                headers=headers,
                params={
                    "node_type": "folder",
                    "limit": min(limit, MAX_PAGE_SIZE),
                    "offset": max(offset, 0)
                },
                timeout=HTTP_TIMEOUT
            )

            if response.status_code in [200, 201]:
                nodes_data = response.json()
                folders = []

                # Handle case where response might be None or not a list
                if nodes_data is None:
                    nodes_data = []
                elif not isinstance(nodes_data, list):
                    nodes_data = [nodes_data]

                # Extract folder data including descriptions for root folders only (parent_id is None)
                for node in nodes_data:
                    if isinstance(node, dict) and node.get("node_type") == "folder" and node.get("parent_id") is None:
                        folder_info = {
                            "id": node.get("id"),
                            "title": node.get("title"),
                            "description": node.get("folder_data", {}).get("description", "") if isinstance(node.get("folder_data"), dict) else ""
                        }
                        folders.append(folder_info)

                return {
                    "success": True,
                    "message": f"Found {len(folders)} root folders",
                    "folders": folders
                }
            else:
                return create_error_response(
                    f"Failed to retrieve root folders: HTTP {response.status_code}",
                    "Check your authentication",
                    "API_ERROR",
                    {"http_status": response.status_code, "response": response.text}
                )

    except httpx.TimeoutException:
        return create_error_response(
            "Request timed out",
            "The FastGTD API is not responding. Check your FASTGTD_API_URL and network connection",
            "TIMEOUT"
        )
    except Exception as e:
        logger.error(f"Failed to get root folders: {str(e)}")
        return create_error_response(
            f"Failed to get root folders: {str(e)}",
            "Check your network connection and API configuration",
            "EXCEPTION"
        )

async def get_root_nodes(limit: int = DEFAULT_PAGE_SIZE, offset: int = 0, auth_token: str = "", current_node_id: str = "") -> dict:
    """Get all root-level nodes (all node types with no parent) - tasks, notes, folders, smart folders, templates"""
    logger.info(f"get_root_nodes called - limit={limit}, offset={offset}")

    # Get auth token if not provided
    if not auth_token:
        auth_token = await get_auth_token()

    if not auth_token:
        return get_error_response("no_auth")

    # FastGTD API endpoint
    url = f"{FASTGTD_API_URL}/nodes/"

    # Prepare headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_token}"
    }

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            # Get root nodes for each type separately
            node_types = ['folder', 'smart_folder', 'template', 'task', 'note']
            all_nodes = []

            for node_type in node_types:
                try:
                    response = await client.get(
                        url,
                        headers=headers,
                        params={
                            "node_type": node_type,
                            "limit": min(limit, MAX_PAGE_SIZE),
                            "offset": max(offset, 0)
                        },
                        timeout=HTTP_TIMEOUT
                    )

                    if response.status_code in [200, 201]:
                        nodes = response.json()

                        # Handle case where response might be None or not a list
                        if nodes is None:
                            nodes = []
                        elif not isinstance(nodes, list):
                            nodes = [nodes]

                        all_nodes.extend(nodes)
                except Exception as e:
                    logger.debug(f"Failed to get {node_type} nodes: {e}")
                    continue

            # Filter to only root level nodes (parent_id is None/null)
            root_nodes = [node for node in all_nodes if isinstance(node, dict) and node.get('parent_id') is None]

            # Sort by node type first, then by title
            root_nodes.sort(key=lambda x: (x.get('node_type', ''), x.get('title', '')))

            return {
                "success": True,
                "message": f"Found {len(root_nodes)} root-level node(s)",
                "root_nodes": [
                    {
                        "id": node.get("id"),
                        "title": node.get("title"),
                        "node_type": node.get("node_type"),
                        "created_at": node.get("created_at"),
                        "updated_at": node.get("updated_at"),
                        "tags": [tag.get("name", "") for tag in node.get("tags", [])]
                    }
                    for node in root_nodes
                ],
                "total_count": len(root_nodes),
                "breakdown": {
                    "folders": len([n for n in root_nodes if n.get('node_type') == 'folder']),
                    "smart_folders": len([n for n in root_nodes if n.get('node_type') == 'smart_folder']),
                    "templates": len([n for n in root_nodes if n.get('node_type') == 'template']),
                    "tasks": len([n for n in root_nodes if n.get('node_type') == 'task']),
                    "notes": len([n for n in root_nodes if n.get('node_type') == 'note'])
                }
            }

    except httpx.TimeoutException:
        return create_error_response(
            "Request timed out",
            "The FastGTD API is not responding. Check your FASTGTD_API_URL and network connection",
            "TIMEOUT"
        )
    except Exception as e:
        logger.error(f"Failed to get root nodes: {str(e)}")
        return create_error_response(
            f"Failed to get root nodes: {str(e)}",
            "Check your network connection and API configuration",
            "EXCEPTION"
        )

async def get_node_children(node_id: str, node_type: str = "", limit: int = DEFAULT_PAGE_SIZE, offset: int = 0, auth_token: str = "", current_node_id: str = "") -> dict:
    """Get immediate children of a specific node (optionally filtered by node type)"""
    logger.info(f"get_node_children called - node_id='{node_id}', node_type='{node_type}'")

    # Validate inputs
    if not node_id or len(node_id.strip()) == 0:
        return get_error_response("node_id_required")

    # Validate node_type if provided
    if node_type:
        valid_types = ["task", "note", "folder", "smart_folder", "template"]
        if node_type not in valid_types:
            return get_error_response("invalid_node_type")

    # Get auth token if not provided
    if not auth_token:
        auth_token = await get_auth_token()

    if not auth_token:
        return get_error_response("no_auth")

    # FastGTD API endpoint
    url = f"{FASTGTD_API_URL}/nodes/"

    # Prepare headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_token}"
    }

    # Prepare query parameters with pagination
    params = {
        "parent_id": node_id.strip(),
        "limit": min(limit, MAX_PAGE_SIZE),
        "offset": max(offset, 0)
    }
    if node_type:
        params["node_type"] = node_type

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.get(
                url,
                headers=headers,
                params=params,
                timeout=HTTP_TIMEOUT
            )

            if response.status_code in [200, 201]:
                nodes_data = response.json()
                children = []

                # Extract child node information
                for node in nodes_data:
                    child_info = {
                        "id": node.get("id"),
                        "title": node.get("title"),
                        "node_type": node.get("node_type"),
                        "created_at": node.get("created_at"),
                        "updated_at": node.get("updated_at")
                    }

                    # Add type-specific data
                    if node.get("node_type") == "task":
                        task_data = node.get("task_data", {})
                        child_info["status"] = task_data.get("status")
                        child_info["priority"] = task_data.get("priority")
                    elif node.get("node_type") == "note":
                        note_data = node.get("note_data", {})
                        child_info["body_preview"] = (note_data.get("body", ""))[:100] + "..." if len(note_data.get("body", "")) > 100 else note_data.get("body", "")

                    children.append(child_info)

                type_filter_msg = f" of type '{node_type}'" if node_type else ""
                return {
                    "success": True,
                    "message": f"Found {len(children)} children{type_filter_msg} for node {node_id}",
                    "children": children,
                    "parent_node_id": node_id,
                    "node_type_filter": node_type
                }
            else:
                return create_error_response(
                    f"Failed to get node children: HTTP {response.status_code}",
                    "Check your authentication and node ID",
                    "API_ERROR",
                    {"http_status": response.status_code, "response": response.text}
                )

    except httpx.TimeoutException:
        return create_error_response(
            "Request timed out",
            "The FastGTD API is not responding. Check your FASTGTD_API_URL and network connection",
            "TIMEOUT"
        )
    except Exception as e:
        logger.error(f"Failed to get node children: {str(e)}")
        return create_error_response(
            f"Failed to get node children: {str(e)}",
            "Check your network connection and API configuration",
            "EXCEPTION"
        )

async def get_folder_id(folder_name: str, auth_token: str = "", current_node_id: str = "") -> dict:
    """Get folder ID by folder name - useful for finding the specific folder to work with"""
    logger.info(f"get_folder_id called - folder_name='{folder_name}'")

    # Validate inputs
    if not folder_name or len(folder_name.strip()) == 0:
        return create_error_response("Folder name is required", "Provide a folder name to search for", "MISSING_FOLDER_NAME")

    # Get auth token if not provided
    if not auth_token:
        auth_token = await get_auth_token()

    if not auth_token:
        return get_error_response("no_auth")

    # FastGTD API endpoint
    url = f"{FASTGTD_API_URL}/nodes/"

    # Prepare headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_token}"
    }

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.get(
                url,
                headers=headers,
                params={"node_type": "folder", "limit": 1000},
                timeout=HTTP_TIMEOUT
            )

            if response.status_code in [200, 201]:
                nodes_data = response.json()

                # Handle case where response might be None or not a list
                if nodes_data is None:
                    nodes_data = []
                elif not isinstance(nodes_data, list):
                    nodes_data = [nodes_data]

                # Find folder with matching name (case-insensitive)
                folder_name_lower = folder_name.lower().strip()
                for node in nodes_data:
                    if isinstance(node, dict) and node.get("node_type") == "folder":
                        node_title = node.get("title", "").lower().strip()
                        if node_title == folder_name_lower:
                            folder_data = node.get("folder_data")
                            result = {
                                "success": True,
                                "message": f"Found folder '{folder_name}'",
                                "folder_id": node.get("id"),
                                "folder_name": node.get("title"),
                                "folder_description": folder_data.get("description", "") if isinstance(folder_data, dict) else ""
                            }
                            logger.debug(f"Found folder_id={result['folder_id']} for '{folder_name}'")
                            return result

                # If exact match not found, check for partial matches
                partial_matches = []
                for node in nodes_data:
                    if isinstance(node, dict) and node.get("node_type") == "folder":
                        node_title = node.get("title", "").lower().strip()
                        if folder_name_lower in node_title or node_title in folder_name_lower:
                            partial_matches.append({
                                "id": node.get("id"),
                                "title": node.get("title")
                            })

                if partial_matches:
                    logger.debug(f"No exact match for '{folder_name}', found {len(partial_matches)} similar")
                    return {
                        "success": False,
                        "error": f"No exact match found for '{folder_name}', but found similar folders",
                        "suggestions": partial_matches
                    }
                else:
                    return create_error_response(
                        f"No folder found with name '{folder_name}'",
                        "Use get_all_folders() to see available folders",
                        "FOLDER_NOT_FOUND"
                    )
            else:
                return create_error_response(
                    f"Failed to retrieve folders: HTTP {response.status_code}",
                    "Check your authentication",
                    "API_ERROR",
                    {"http_status": response.status_code, "response": response.text}
                )

    except httpx.TimeoutException:
        return create_error_response(
            "Request timed out",
            "The FastGTD API is not responding. Check your FASTGTD_API_URL and network connection",
            "TIMEOUT"
        )
    except Exception as e:
        logger.error(f"Failed to find folder: {str(e)}")
        return create_error_response(
            f"Failed to find folder: {str(e)}",
            "Check your network connection and API configuration",
            "EXCEPTION"
        )

async def add_task_to_node_id(node_id: str, task_title: str, description: str = "", priority: str = "medium", auth_token: str = "", current_node_id: str = "") -> dict:
    """Add a task to a specific node by node ID and return the new task's ID"""
    logger.info(f"add_task_to_node_id called - node_id='{node_id}', task_title='{task_title}'")

    # Validate inputs
    if not node_id:
        return get_error_response("node_id_required")

    if not task_title or len(task_title.strip()) == 0:
        return create_error_response("Task title is required", "Provide a task title", "MISSING_TITLE")

    # Validate priority
    valid_priorities = ["low", "medium", "high", "urgent"]
    if priority and priority not in valid_priorities:
        return get_error_response("invalid_priority")

    # Get auth token if not provided
    if not auth_token:
        auth_token = await get_auth_token()

    if not auth_token:
        return get_error_response("no_auth")

    # FastGTD API endpoint
    url = f"{FASTGTD_API_URL}/nodes/"

    # Create task payload for unified node system
    task_payload = {
        "node_type": "task",
        "title": task_title.strip(),
        "parent_id": node_id,
        "task_data": {
            "description": description,
            "priority": priority,
            "status": "todo",
            "archived": False
        }
    }

    # Prepare headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_token}"
    }

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.post(
                url,
                json=task_payload,
                headers=headers,
                timeout=HTTP_TIMEOUT
            )

            if response.status_code in [200, 201]:
                task_data = response.json()
                task_id = task_data.get("id")
                logger.info(f"add_task_to_node_id success - created task_id={task_id} in node_id={node_id}")
                return {
                    "success": True,
                    "message": f"Task '{task_title}' added to node successfully",
                    "task_id": task_id,
                    "node_id": node_id,
                    "task_title": task_title
                }
            else:
                logger.error(f"add_task_to_node_id failed - HTTP {response.status_code}")
                return create_error_response(
                    f"Failed to add task: HTTP {response.status_code}",
                    "Check your authentication and node ID",
                    "API_ERROR",
                    {"http_status": response.status_code, "response": response.text}
                )

    except httpx.TimeoutException:
        return create_error_response(
            "Request timed out",
            "The FastGTD API is not responding. Check your FASTGTD_API_URL and network connection",
            "TIMEOUT"
        )
    except Exception as e:
        logger.error(f"Failed to add task to node: {str(e)}")
        return create_error_response(
            f"Failed to add task to node: {str(e)}",
            "Check your network connection and API configuration",
            "EXCEPTION"
        )

async def get_node_tree(root_id: str = "", max_depth: int = 10, auth_token: str = "", current_node_id: str = "") -> dict:
    """Get the node tree structure starting from a root node (or from root if no ID provided)"""
    logger.info(f"get_node_tree called - root_id='{root_id}', max_depth={max_depth}")

    # Validate max_depth
    if max_depth < 1 or max_depth > MAX_TREE_DEPTH:
        return create_error_response(
            f"max_depth must be between 1 and {MAX_TREE_DEPTH}",
            f"Use a value between 1 and {MAX_TREE_DEPTH}",
            "INVALID_DEPTH"
        )

    # Get auth token if not provided
    if not auth_token:
        auth_token = await get_auth_token()

    if not auth_token:
        return get_error_response("no_auth")

    # FastGTD API endpoint for tree
    if root_id:
        url = f"{FASTGTD_API_URL}/nodes/tree/{root_id.strip()}"
    else:
        url = f"{FASTGTD_API_URL}/nodes/"

    # Prepare headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_token}"
    }

    # Query parameters
    params = {}
    if root_id:
        params["max_depth"] = max_depth
    else:
        params["limit"] = 100

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.get(
                url,
                headers=headers,
                params=params,
                timeout=HTTP_TIMEOUT
            )

            if response.status_code == 200:
                data = response.json()

                if root_id:
                    result = {
                        "success": True,
                        "message": f"Retrieved node tree from {root_id} (depth: {max_depth})",
                        "tree": data,
                        "root_id": root_id,
                        "item_count": data.get("total_count", 0),
                        "format": "tree"
                    }
                    logger.debug(f"Retrieved tree with {result['item_count']} items")
                else:
                    result = {
                        "success": True,
                        "message": "Retrieved root level nodes",
                        "nodes": data,
                        "root_id": "root",
                        "item_count": len(data) if isinstance(data, list) else 1,
                        "format": "list"
                    }
                    logger.debug(f"Retrieved {result['item_count']} root nodes")

                return result
            else:
                return create_error_response(
                    f"Failed to get node tree: HTTP {response.status_code}",
                    "Check your authentication and node ID",
                    "API_ERROR",
                    {"http_status": response.status_code, "response": response.text}
                )

    except httpx.TimeoutException:
        return create_error_response(
            "Request timed out",
            "The FastGTD API is not responding. Check your FASTGTD_API_URL and network connection",
            "TIMEOUT"
        )
    except Exception as e:
        logger.error(f"Failed to get node tree: {str(e)}")
        return create_error_response(
            f"Failed to get node tree: {str(e)}",
            "Check your network connection and API configuration",
            "EXCEPTION"
        )

async def search_nodes(query: str, node_type: str = "", limit: int = 50, offset: int = 0, auth_token: str = "", current_node_id: str = "") -> dict:
    """Search for nodes by title and content - perfect for finding specific tasks, notes, or folders"""
    logger.info(f"search_nodes called - query='{query}', node_type='{node_type}', limit={limit}")

    # Validate inputs
    if not query or len(query.strip()) < 1:
        return get_error_response("search_query_required")

    # Validate node type if specified
    if node_type and node_type not in ["task", "note", "folder", "smart_folder", "template"]:
        return get_error_response("invalid_node_type")

    # Get auth token if not provided
    if not auth_token:
        auth_token = await get_auth_token()

    if not auth_token:
        return get_error_response("no_auth")

    # FastGTD API endpoint for searching nodes
    url = f"{FASTGTD_API_URL}/nodes/"

    # Prepare headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_token}"
    }

    # Query parameters for search
    params = {
        "search": query.strip(),
        "limit": min(limit, MAX_PAGE_SIZE),
        "offset": max(offset, 0)
    }

    # Add node type filter if specified
    if node_type:
        params["node_type"] = node_type

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.get(
                url,
                headers=headers,
                params=params,
                timeout=HTTP_TIMEOUT
            )

            if response.status_code == 200:
                results = response.json()
                result_count = len(results) if isinstance(results, list) else 1

                return {
                    "success": True,
                    "message": f"Found {result_count} result(s) for '{query}'",
                    "query": query,
                    "node_type_filter": node_type or "all types",
                    "results": results,
                    "result_count": result_count
                }
            else:
                return create_error_response(
                    f"Search failed: HTTP {response.status_code}",
                    "Check your authentication and parameters",
                    "API_ERROR",
                    {"http_status": response.status_code, "response": response.text}
                )

    except httpx.TimeoutException:
        return create_error_response(
            "Request timed out",
            "The FastGTD API is not responding. Check your FASTGTD_API_URL and network connection",
            "TIMEOUT"
        )
    except Exception as e:
        logger.error(f"search_nodes failed: {str(e)}")
        return create_error_response(
            f"search_nodes failed: {str(e)}",
            "Check your network connection and API configuration",
            "EXCEPTION"
        )

async def create_task(title: str, description: str = "", priority: str = "medium", parent_id: str = "", auth_token: str = "", current_node_id: str = "") -> dict:
    """Create a new task with simplified interface - auto-detects best location"""
    logger.info(f"create_task called - title='{title}', priority='{priority}', parent_id='{parent_id}'")

    # Validate inputs
    if not title or len(title.strip()) == 0:
        return create_error_response("Title is required", "Provide a task title", "MISSING_TITLE")

    # Validate priority
    if priority and priority not in ["low", "medium", "high", "urgent"]:
        return get_error_response("invalid_priority")

    # Get auth token if not provided
    if not auth_token:
        auth_token = await get_auth_token()

    if not auth_token:
        return get_error_response("no_auth")

    # FastGTD API endpoint
    url = f"{FASTGTD_API_URL}/nodes/"

    # Create task payload for unified node system
    task_payload = {
        "node_type": "task",
        "title": title.strip(),
        "task_data": {
            "description": description,
            "priority": priority,
            "status": "todo",
            "archived": False
        }
    }

    # Determine parent location
    if parent_id:
        task_payload["parent_id"] = parent_id
    elif current_node_id:
        task_payload["parent_id"] = current_node_id
        logger.debug(f"Using current node as parent: {current_node_id}")

    # Prepare headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_token}"
    }

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            # If no parent specified, get default node
            if not parent_id and not current_node_id:
                default_node_response = await client.get(
                    f"{FASTGTD_API_URL}/settings/default-node",
                    headers=headers,
                    timeout=HTTP_TIMEOUT
                )

                if default_node_response.status_code == 200:
                    default_data = default_node_response.json()
                    default_node_id = default_data.get("node_id")
                    if default_node_id:
                        task_payload["parent_id"] = default_node_id
                        logger.debug(f"Using default node as parent: {default_node_id}")

            # Create the task
            response = await client.post(
                url,
                json=task_payload,
                headers=headers,
                timeout=HTTP_TIMEOUT
            )

            if response.status_code in [200, 201]:
                task_data = response.json()
                return {
                    "success": True,
                    "message": f"Task '{title}' created successfully",
                    "task_id": task_data.get("id"),
                    "task": task_data
                }
            else:
                return create_error_response(
                    f"Failed to create task: HTTP {response.status_code}",
                    "Check your authentication and parameters",
                    "API_ERROR",
                    {"http_status": response.status_code, "response": response.text}
                )

    except httpx.TimeoutException:
        return create_error_response(
            "Request timed out",
            "The FastGTD API is not responding. Check your FASTGTD_API_URL and network connection",
            "TIMEOUT"
        )
    except Exception as e:
        logger.error(f"create_task failed: {str(e)}")
        return create_error_response(
            f"create_task failed: {str(e)}",
            "Check your network connection and API configuration",
            "EXCEPTION"
        )

async def update_task(task_id: str, title: str = "", description: str = "", priority: str = "", status: str = "", due_at: str = "", earliest_start_at: str = "", archived: bool = None, recurrence_rule: str = "", recurrence_anchor: str = "", auth_token: str = "", current_node_id: str = "") -> dict:
    """Update an existing task's properties"""
    logger.info(f"update_task called - task_id='{task_id}'")

    # Validate inputs
    if not task_id or len(task_id.strip()) == 0:
        return get_error_response("node_id_required")

    # Validate priority if specified
    if priority and priority not in ["low", "medium", "high", "urgent"]:
        return get_error_response("invalid_priority")

    # Get auth token if not provided
    if not auth_token:
        auth_token = await get_auth_token()

    if not auth_token:
        return get_error_response("no_auth")

    # FastGTD API endpoint
    url = f"{FASTGTD_API_URL}/nodes/{task_id}"

    # Build update payload - only include fields that are provided
    update_payload = {}

    if title:
        update_payload["title"] = title.strip()

    # Task-specific data updates
    task_data_updates = {}
    if description:
        task_data_updates["description"] = description
    if priority:
        task_data_updates["priority"] = priority
    if status:
        task_data_updates["status"] = status
    if due_at:
        task_data_updates["due_at"] = due_at
    if earliest_start_at:
        task_data_updates["earliest_start_at"] = earliest_start_at
    if archived is not None:
        task_data_updates["archived"] = archived
    if recurrence_rule:
        task_data_updates["recurrence_rule"] = recurrence_rule
    if recurrence_anchor:
        task_data_updates["recurrence_anchor"] = recurrence_anchor

    if task_data_updates:
        update_payload["task_data"] = task_data_updates

    # Check if we have anything to update
    if not update_payload:
        return create_error_response(
            "No fields provided to update",
            "Specify at least one field: title, description, priority, status, due_at, earliest_start_at, archived, recurrence_rule, or recurrence_anchor",
            "MISSING_FIELDS"
        )

    # Prepare headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_token}"
    }

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.put(
                url,
                json=update_payload,
                headers=headers,
                timeout=HTTP_TIMEOUT
            )

            if response.status_code == 200:
                task_data = response.json()
                return {
                    "success": True,
                    "message": f"Task '{task_id}' updated successfully",
                    "task": {
                        "id": task_data.get("id"),
                        "title": task_data.get("title"),
                        "description": task_data.get("task_data", {}).get("description", ""),
                        "priority": task_data.get("task_data", {}).get("priority", ""),
                        "status": task_data.get("task_data", {}).get("status", ""),
                        "updated_fields": list(update_payload.keys())
                    }
                }
            else:
                return create_error_response(
                    f"Failed to update task: HTTP {response.status_code}",
                    "Check your authentication and parameters",
                    "API_ERROR",
                    {"http_status": response.status_code, "response": response.text}
                )

    except httpx.TimeoutException:
        return create_error_response(
            "Request timed out",
            "The FastGTD API is not responding. Check your FASTGTD_API_URL and network connection",
            "TIMEOUT"
        )
    except Exception as e:
        logger.error(f"update_task failed: {str(e)}")
        return create_error_response(
            f"update_task failed: {str(e)}",
            "Check your network connection and API configuration",
            "EXCEPTION"
        )

async def complete_task(task_id: str, auth_token: str = "", current_node_id: str = "") -> dict:
    """Mark a task as completed"""
    logger.info(f"complete_task called - task_id='{task_id}'")

    # Validate inputs
    if not task_id or len(task_id.strip()) == 0:
        return get_error_response("node_id_required")

    # Get auth token if not provided
    if not auth_token:
        auth_token = await get_auth_token()

    if not auth_token:
        return get_error_response("no_auth")

    # FastGTD API endpoint
    url = f"{FASTGTD_API_URL}/nodes/{task_id}"

    # Update payload to mark as done
    update_payload = {
        "task_data": {
            "status": "done"
        }
    }

    # Prepare headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_token}"
    }

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.put(
                url,
                json=update_payload,
                headers=headers,
                timeout=HTTP_TIMEOUT
            )

            if response.status_code == 200:
                task_data = response.json()
                return {
                    "success": True,
                    "message": f"Task '{task_id}' marked as completed",
                    "task": {
                        "id": task_data.get("id"),
                        "title": task_data.get("title"),
                        "status": task_data.get("task_data", {}).get("status", ""),
                    }
                }
            else:
                return create_error_response(
                    f"Failed to complete task: HTTP {response.status_code}",
                    "Check your authentication and parameters",
                    "API_ERROR",
                    {"http_status": response.status_code, "response": response.text}
                )

    except httpx.TimeoutException:
        return create_error_response(
            "Request timed out",
            "The FastGTD API is not responding. Check your FASTGTD_API_URL and network connection",
            "TIMEOUT"
        )
    except Exception as e:
        logger.error(f"complete_task failed: {str(e)}")
        return create_error_response(
            f"complete_task failed: {str(e)}",
            "Check your network connection and API configuration",
            "EXCEPTION"
        )

async def delete_folder(folder_id: str, auth_token: str = "", current_node_id: str = "") -> dict:
    """Delete a folder permanently (will also delete all contents)"""
    logger.info(f"delete_folder called - folder_id='{folder_id}'")

    # Validate inputs
    if not folder_id or len(folder_id.strip()) == 0:
        return get_error_response("node_id_required")

    # Get auth token if not provided
    if not auth_token:
        auth_token = await get_auth_token()

    if not auth_token:
        return get_error_response("no_auth")

    # FastGTD API endpoint
    url = f"{FASTGTD_API_URL}/nodes/{folder_id}"

    # Prepare headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_token}"
    }

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.delete(
                url,
                headers=headers,
                timeout=HTTP_TIMEOUT
            )

            if response.status_code in [204, 200]:
                return {
                    "success": True,
                    "message": "Folder deleted successfully",
                    "folder_id": folder_id
                }
            elif response.status_code == 404:
                return create_error_response(
                    "Folder not found",
                    "Check the folder ID and try again",
                    "NOT_FOUND"
                )
            elif response.status_code == 403:
                return create_error_response(
                    "Permission denied",
                    "You do not have permission to delete this folder",
                    "FORBIDDEN"
                )
            else:
                return create_error_response(
                    f"Failed to delete folder: HTTP {response.status_code}",
                    "Check your authentication and parameters",
                    "API_ERROR",
                    {"http_status": response.status_code, "response": response.text}
                )

    except httpx.TimeoutException:
        return create_error_response(
            "Request timed out",
            "The FastGTD API is not responding. Check your FASTGTD_API_URL and network connection",
            "TIMEOUT"
        )
    except Exception as e:
        logger.error(f"delete_folder failed: {str(e)}")
        return create_error_response(
            f"delete_folder failed: {str(e)}",
            "Check your network connection and API configuration",
            "EXCEPTION"
        )

async def delete_task(task_id: str, auth_token: str = "", current_node_id: str = "") -> dict:
    """Delete a task permanently"""
    logger.info(f"delete_task called - task_id='{task_id}'")

    # Validate inputs
    if not task_id or len(task_id.strip()) == 0:
        return get_error_response("node_id_required")

    # Get auth token if not provided
    if not auth_token:
        auth_token = await get_auth_token()

    if not auth_token:
        return get_error_response("no_auth")

    # FastGTD API endpoint
    url = f"{FASTGTD_API_URL}/nodes/{task_id}"

    # Prepare headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_token}"
    }

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.delete(
                url,
                headers=headers,
                timeout=HTTP_TIMEOUT
            )

            if response.status_code in [200, 204]:
                return {
                    "success": True,
                    "message": f"Task '{task_id}' deleted successfully"
                }
            else:
                return create_error_response(
                    f"Failed to delete task: HTTP {response.status_code}",
                    "Check your authentication and parameters",
                    "API_ERROR",
                    {"http_status": response.status_code, "response": response.text}
                )

    except httpx.TimeoutException:
        return create_error_response(
            "Request timed out",
            "The FastGTD API is not responding. Check your FASTGTD_API_URL and network connection",
            "TIMEOUT"
        )
    except Exception as e:
        logger.error(f"delete_task failed: {str(e)}")
        return create_error_response(
            f"delete_task failed: {str(e)}",
            "Check your network connection and API configuration",
            "EXCEPTION"
        )

async def create_folder(title: str, description: str = "", parent_id: str = "", auth_token: str = "", current_node_id: str = "") -> dict:
    """Create a new folder - auto-detects best location (current folder or root)"""
    logger.info(f"create_folder called - title='{title}', parent_id='{parent_id}'")

    # Validate inputs
    if not title or len(title.strip()) == 0:
        return create_error_response("Title is required", "Provide a folder title", "MISSING_TITLE")

    # Get auth token if not provided
    if not auth_token:
        auth_token = await get_auth_token()

    if not auth_token:
        return get_error_response("no_auth")

    # FastGTD API endpoint
    url = f"{FASTGTD_API_URL}/nodes/"

    # Create folder payload
    folder_payload = {
        "node_type": "folder",
        "title": title.strip()
    }

    # Add description if provided
    if description:
        folder_payload["folder_data"] = {"description": description}

    # Determine parent location
    if parent_id:
        folder_payload["parent_id"] = parent_id
    elif current_node_id:
        folder_payload["parent_id"] = current_node_id
        logger.debug(f"Using current node as parent: {current_node_id}")

    # Prepare headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_token}"
    }

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            # If no parent specified, get default node
            if not parent_id and not current_node_id:
                default_node_response = await client.get(
                    f"{FASTGTD_API_URL}/settings/default-node",
                    headers=headers,
                    timeout=HTTP_TIMEOUT
                )

                if default_node_response.status_code == 200:
                    default_data = default_node_response.json()
                    default_node_id = default_data.get("node_id")
                    if default_node_id:
                        folder_payload["parent_id"] = default_node_id
                        logger.debug(f"Using default node as parent: {default_node_id}")

            # Create the folder
            response = await client.post(
                url,
                json=folder_payload,
                headers=headers,
                timeout=HTTP_TIMEOUT
            )

            if response.status_code in [200, 201]:
                folder_data = response.json()
                return {
                    "success": True,
                    "message": f"Folder '{title}' created successfully",
                    "folder": {
                        "id": folder_data.get("id"),
                        "title": folder_data.get("title"),
                        "parent_id": folder_data.get("parent_id")
                    }
                }
            else:
                return create_error_response(
                    f"Failed to create folder: HTTP {response.status_code}",
                    "Check your authentication and parameters",
                    "API_ERROR",
                    {"http_status": response.status_code, "response": response.text}
                )

    except httpx.TimeoutException:
        return create_error_response(
            "Request timed out",
            "The FastGTD API is not responding. Check your FASTGTD_API_URL and network connection",
            "TIMEOUT"
        )
    except Exception as e:
        logger.error(f"create_folder failed: {str(e)}")
        return create_error_response(
            f"create_folder failed: {str(e)}",
            "Check your network connection and API configuration",
            "EXCEPTION"
        )

async def move_node(node_id: str, new_parent_id: str = "", new_sort_order: int = None, auth_token: str = "", current_node_id: str = "") -> dict:
    """Move a task or note to a different folder (or to root if no parent specified)"""
    logger.info(f"move_node called - node_id='{node_id}', new_parent_id='{new_parent_id}'")

    # Validate inputs
    if not node_id or len(node_id.strip()) == 0:
        return get_error_response("node_id_required")

    # Get auth token if not provided
    if not auth_token:
        auth_token = await get_auth_token()

    if not auth_token:
        return get_error_response("no_auth")

    # FastGTD API endpoint for moving nodes
    url = f"{FASTGTD_API_URL}/nodes/move"

    # Create move payload
    move_payload = {
        "node_id": node_id.strip(),
        "new_parent_id": new_parent_id if new_parent_id else None
    }

    if new_sort_order is not None:
        move_payload["new_sort_order"] = new_sort_order

    # Prepare headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_token}"
    }

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.post(
                url,
                json=move_payload,
                headers=headers,
                timeout=HTTP_TIMEOUT
            )

            if response.status_code in [200, 201]:
                result_data = response.json() if response.content else {"message": "Node moved successfully"}
                parent_msg = f"to folder {new_parent_id}" if new_parent_id else "to root level"

                return {
                    "success": True,
                    "message": f"Node {node_id} moved {parent_msg} successfully",
                    "node_id": node_id,
                    "new_parent_id": new_parent_id,
                    "new_sort_order": new_sort_order,
                    "api_response": result_data
                }
            else:
                return create_error_response(
                    f"Failed to move node: HTTP {response.status_code}",
                    "Check your authentication and parameters",
                    "API_ERROR",
                    {"http_status": response.status_code, "response": response.text}
                )

    except httpx.TimeoutException:
        return create_error_response(
            "Request timed out",
            "The FastGTD API is not responding. Check your FASTGTD_API_URL and network connection",
            "TIMEOUT"
        )
    except Exception as e:
        logger.error(f"move_node failed: {str(e)}")
        return create_error_response(
            f"move_node failed: {str(e)}",
            "Check your network connection and API configuration",
            "EXCEPTION"
        )

async def add_tag(node_id: str, tag_name: str, tag_description: str = "", tag_color: str = "", auth_token: str = "", current_node_id: str = "") -> dict:
    """Add a tag to a task, note, or folder (creates tag if it doesn't exist)"""
    logger.info(f"add_tag called - node_id='{node_id}', tag_name='{tag_name}'")

    # Validate inputs
    if not node_id or len(node_id.strip()) == 0:
        return get_error_response("node_id_required")

    if not tag_name or len(tag_name.strip()) == 0:
        return create_error_response("Tag name is required", "Provide a tag name", "MISSING_TAG_NAME")

    # Get auth token if not provided
    if not auth_token:
        auth_token = await get_auth_token()

    if not auth_token:
        return get_error_response("no_auth")

    # FastGTD API endpoints
    create_tag_url = f"{FASTGTD_API_URL}/tags"
    attach_tag_url = f"{FASTGTD_API_URL}/nodes/{node_id}/tags"

    # Prepare headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_token}"
    }

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            # Step 1: Create or get tag
            create_body = {"name": tag_name.strip()}
            if tag_description:
                create_body["description"] = tag_description
            if tag_color:
                create_body["color"] = tag_color

            tag_response = await client.post(
                create_tag_url,
                headers=headers,
                json=create_body,
                timeout=HTTP_TIMEOUT
            )

            if tag_response.status_code not in [200, 201]:
                return create_error_response(
                    f"Failed to create tag: HTTP {tag_response.status_code}",
                    "Check your authentication and parameters",
                    "API_ERROR",
                    {"http_status": tag_response.status_code, "response": tag_response.text}
                )

            tag_data = tag_response.json()
            tag_id = tag_data.get("id")

            if not tag_id:
                return create_error_response(
                    "Failed to get tag ID from response",
                    "The tag may not have been created properly",
                    "MISSING_TAG_ID"
                )

            # Step 2: Attach tag to node
            attach_url = f"{attach_tag_url}/{tag_id}"
            attach_response = await client.post(
                attach_url,
                headers=headers,
                timeout=HTTP_TIMEOUT
            )

            if attach_response.status_code in [200, 201]:
                return {
                    "success": True,
                    "message": f"Tag '{tag_name}' added to node successfully",
                    "node_id": node_id,
                    "tag": {
                        "id": tag_id,
                        "name": tag_name,
                        "description": tag_description,
                        "color": tag_color,
                        "existed_before": tag_data.get("existed", False)
                    }
                }
            else:
                return create_error_response(
                    f"Failed to attach tag to node: HTTP {attach_response.status_code}",
                    "Check your authentication and parameters",
                    "API_ERROR",
                    {"http_status": attach_response.status_code, "response": attach_response.text}
                )

    except httpx.TimeoutException:
        return create_error_response(
            "Request timed out",
            "The FastGTD API is not responding. Check your FASTGTD_API_URL and network connection",
            "TIMEOUT"
        )
    except Exception as e:
        logger.error(f"add_tag failed: {str(e)}")
        return create_error_response(
            f"add_tag failed: {str(e)}",
            "Check your network connection and API configuration",
            "EXCEPTION"
        )

async def remove_tag(node_id: str, tag_name: str, auth_token: str = "", current_node_id: str = "") -> dict:
    """Remove a tag from a task, note, or folder"""
    logger.info(f"remove_tag called - node_id='{node_id}', tag_name='{tag_name}'")

    # Validate inputs
    if not node_id or len(node_id.strip()) == 0:
        return get_error_response("node_id_required")

    if not tag_name or len(tag_name.strip()) == 0:
        return create_error_response("Tag name is required", "Provide a tag name", "MISSING_TAG_NAME")

    # Get auth token if not provided
    if not auth_token:
        auth_token = await get_auth_token()

    if not auth_token:
        return get_error_response("no_auth")

    # FastGTD API endpoints
    get_tags_url = f"{FASTGTD_API_URL}/nodes/{node_id}/tags"
    detach_tag_url = f"{FASTGTD_API_URL}/nodes/{node_id}/tags"

    # Prepare headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_token}"
    }

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            # Step 1: Get all tags for the node to find the tag ID
            tags_response = await client.get(
                get_tags_url,
                headers=headers,
                timeout=HTTP_TIMEOUT
            )

            if tags_response.status_code != 200:
                return create_error_response(
                    f"Failed to get node tags: HTTP {tags_response.status_code}",
                    "Check your authentication and parameters",
                    "API_ERROR",
                    {"http_status": tags_response.status_code, "response": tags_response.text}
                )

            tags_data = tags_response.json()

            # Handle case where response might be None or not a list
            if tags_data is None:
                tags_data = []
            elif not isinstance(tags_data, list):
                tags_data = [tags_data]

            # Find the tag with matching name (case-insensitive)
            tag_id = None
            tag_name_lower = tag_name.lower().strip()

            for tag in tags_data:
                if not isinstance(tag, dict):
                    continue
                if tag.get("name", "").lower().strip() == tag_name_lower:
                    tag_id = tag.get("id")
                    break

            if not tag_id:
                return create_error_response(
                    f"Tag '{tag_name}' not found on this node",
                    f"Available tags: {', '.join([tag.get('name') for tag in tags_data if tag.get('name')])}",
                    "TAG_NOT_FOUND",
                    {"available_tags": [tag.get("name") for tag in tags_data if tag.get("name")]}
                )

            # Step 2: Remove tag from node
            detach_url = f"{detach_tag_url}/{tag_id}"
            detach_response = await client.delete(
                detach_url,
                headers=headers,
                timeout=HTTP_TIMEOUT
            )

            if detach_response.status_code in [204, 200]:
                return {
                    "success": True,
                    "message": f"Tag '{tag_name}' removed from node successfully",
                    "node_id": node_id,
                    "tag_name": tag_name,
                    "tag_id": tag_id
                }
            else:
                return create_error_response(
                    f"Failed to remove tag from node: HTTP {detach_response.status_code}",
                    "Check your authentication and parameters",
                    "API_ERROR",
                    {"http_status": detach_response.status_code, "response": detach_response.text}
                )

    except httpx.TimeoutException:
        return create_error_response(
            "Request timed out",
            "The FastGTD API is not responding. Check your FASTGTD_API_URL and network connection",
            "TIMEOUT"
        )
    except Exception as e:
        logger.error(f"remove_tag failed: {str(e)}")
        return create_error_response(
            f"remove_tag failed: {str(e)}",
            "Check your network connection and API configuration",
            "EXCEPTION"
        )


async def get_today_tasks(auth_token: str = "", current_node_id: str = "") -> dict:
    """Get all tasks that are due today."""
    logger.info("get_today_tasks called")

    # Get auth token if not provided
    if not auth_token:
        auth_token = await get_auth_token()

    if not auth_token:
        return get_error_response("no_auth")

    # Get today's date in ISO format (start and end of day)
    from datetime import datetime, timezone
    today = datetime.now(timezone.utc).date()
    today_start = datetime.combine(today, datetime.min.time()).replace(tzinfo=timezone.utc).isoformat()
    today_end = datetime.combine(today, datetime.max.time()).replace(tzinfo=timezone.utc).isoformat()

    # FastGTD API endpoint for getting nodes
    url = f"{FASTGTD_API_URL}/nodes/"

    # Prepare headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_token}"
    }

    # Query parameters - get all tasks and filter by due date on client side
    params = {
        "node_type": "task",
        "limit": 1000  # High limit to ensure we get all tasks
    }

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.get(
                url,
                headers=headers,
                params=params,
                timeout=HTTP_TIMEOUT
            )

            if response.status_code == 200:
                all_tasks = response.json()

                # Handle case where response might be None or not a list
                if all_tasks is None:
                    all_tasks = []
                elif not isinstance(all_tasks, list):
                    all_tasks = [all_tasks]

                today_tasks = []

                for task in all_tasks:
                    if not isinstance(task, dict):
                        continue
                    # Check if task has due_at field and if it's today
                    task_data = task.get('task_data', {})
                    due_at = task_data.get('due_at')

                    if due_at:
                        # Parse due date and check if it's today
                        try:
                            due_date = datetime.fromisoformat(due_at.replace('Z', '+00:00'))
                            if due_date.date() == today:
                                today_tasks.append({
                                    'id': task['id'],
                                    'title': task['title'],
                                    'status': task_data.get('status', 'todo'),
                                    'priority': task_data.get('priority', 'medium'),
                                    'due_at': due_at,
                                    'description': task_data.get('description', '')
                                })
                        except ValueError:
                            # Skip tasks with invalid date format
                            continue

                return {
                    "success": True,
                    "message": f"Found {len(today_tasks)} task(s) due today",
                    "tasks": today_tasks,
                    "today_date": today.isoformat()
                }
            elif response.status_code == 500:
                # API returns 500 when there are issues with task retrieval
                # Return empty list to be resilient
                logger.warning("API returned HTTP 500, returning empty task list")
                return {
                    "success": True,
                    "message": "API error encountered, returning empty task list (API may have data issues)",
                    "tasks": [],
                    "today_date": today.isoformat(),
                    "warning": "API returned HTTP 500"
                }
            else:
                if response.status_code == 401:
                    return get_error_response("no_auth")
                elif response.status_code == 404:
                    return create_error_response(
                        "Tasks endpoint not found",
                        "Verify your FASTGTD_API_URL is correct",
                        "NOT_FOUND"
                    )
                else:
                    return create_error_response(
                        f"API request failed: HTTP {response.status_code}",
                        "Check your network connection and API configuration",
                        "API_ERROR"
                    )

    except httpx.TimeoutException:
        return create_error_response(
            "Request timed out",
            "The FastGTD API is not responding. Check your FASTGTD_API_URL and network connection",
            "TIMEOUT"
        )
    except Exception as e:
        logger.error(f"get_today_tasks failed: {str(e)}")
        return create_error_response(
            f"Failed to get today's tasks: {str(e)}",
            "Check your network connection and API configuration",
            "EXCEPTION"
        )


async def get_overdue_tasks(auth_token: str = "", current_node_id: str = "") -> dict:
    """Get all tasks that are overdue (due date in the past)."""
    logger.info("get_overdue_tasks called")

    # Get auth token if not provided
    if not auth_token:
        auth_token = await get_auth_token()

    if not auth_token:
        return get_error_response("no_auth")

    # Get current datetime
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)

    # FastGTD API endpoint for getting nodes
    url = f"{FASTGTD_API_URL}/nodes/"

    # Prepare headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_token}"
    }

    # Query parameters - get all tasks and filter by due date on client side
    params = {
        "node_type": "task",
        "limit": 1000  # High limit to ensure we get all tasks
    }

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.get(
                url,
                headers=headers,
                params=params,
                timeout=HTTP_TIMEOUT
            )

            if response.status_code == 200:
                all_tasks = response.json()

                # Handle case where response might be None or not a list
                if all_tasks is None:
                    all_tasks = []
                elif not isinstance(all_tasks, list):
                    all_tasks = [all_tasks]

                overdue_tasks = []

                for task in all_tasks:
                    if not isinstance(task, dict):
                        continue
                    # Check if task has due_at field and if it's overdue
                    task_data = task.get('task_data', {})
                    due_at = task_data.get('due_at')
                    status = task_data.get('status', 'todo')

                    # Only include tasks that are not completed and have a due date
                    if due_at and status not in ['done', 'completed']:
                        try:
                            due_date = datetime.fromisoformat(due_at.replace('Z', '+00:00'))
                            if due_date < now:
                                overdue_tasks.append({
                                    'id': task['id'],
                                    'title': task['title'],
                                    'status': status,
                                    'priority': task_data.get('priority', 'medium'),
                                    'due_at': due_at,
                                    'description': task_data.get('description', ''),
                                    'days_overdue': (now.date() - due_date.date()).days
                                })
                        except ValueError:
                            # Skip tasks with invalid date format
                            continue

                # Sort by most overdue first
                overdue_tasks.sort(key=lambda x: x['days_overdue'], reverse=True)

                return {
                    "success": True,
                    "message": f"Found {len(overdue_tasks)} overdue task(s)",
                    "tasks": overdue_tasks,
                    "current_time": now.isoformat()
                }
            elif response.status_code == 500:
                # API returns 500 when there are issues with task retrieval
                # Return empty list to be resilient
                logger.warning("API returned HTTP 500, returning empty task list")
                return {
                    "success": True,
                    "message": "API error encountered, returning empty task list (API may have data issues)",
                    "tasks": [],
                    "current_time": now.isoformat(),
                    "warning": "API returned HTTP 500"
                }
            else:
                if response.status_code == 401:
                    return get_error_response("no_auth")
                elif response.status_code == 404:
                    return create_error_response(
                        "Tasks endpoint not found",
                        "Verify your FASTGTD_API_URL is correct",
                        "NOT_FOUND"
                    )
                else:
                    return create_error_response(
                        f"API request failed: HTTP {response.status_code}",
                        "Check your network connection and API configuration",
                        "API_ERROR"
                    )

    except httpx.TimeoutException:
        return create_error_response(
            "Request timed out",
            "The FastGTD API is not responding. Check your FASTGTD_API_URL and network connection",
            "TIMEOUT"
        )
    except Exception as e:
        logger.error(f"get_overdue_tasks failed: {str(e)}")
        return create_error_response(
            f"Failed to get overdue tasks: {str(e)}",
            "Check your network connection and API configuration",
            "EXCEPTION"
        )


async def update_note(note_id: str, title: str = "", content: str = "", auth_token: str = "", current_node_id: str = "") -> dict:
    """Update an existing note's title and/or content."""
    logger.info(f"update_note called - note_id={note_id}")

    # Get auth token if not provided
    if not auth_token:
        auth_token = await get_auth_token()

    if not auth_token:
        return get_error_response("no_auth")

    if not note_id:
        return get_error_response("node_id_required")

    # Must provide either title or content to update
    if not title and not content:
        return create_error_response(
            "Must provide either title or content to update",
            "Provide at least one field to update: title or content",
            "MISSING_UPDATE_FIELDS"
        )

    # FastGTD API endpoint for updating nodes
    url = f"{FASTGTD_API_URL}/nodes/{note_id}"

    # Prepare headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_token}"
    }

    # Build update payload
    update_data = {}

    # Add title if provided
    if title:
        update_data["title"] = title

    # Add note content if provided
    if content:
        update_data["note_data"] = {
            "body": content
        }

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.put(
                url,
                headers=headers,
                json=update_data,
                timeout=HTTP_TIMEOUT
            )

            if response.status_code == 200:
                updated_note = response.json()

                return {
                    "success": True,
                    "message": f"Note '{updated_note.get('title', 'Unknown')}' updated successfully",
                    "note": {
                        "id": updated_note["id"],
                        "title": updated_note["title"],
                        "content": updated_note.get("note_data", {}).get("body", ""),
                        "updated_at": updated_note["updated_at"]
                    }
                }
            else:
                if response.status_code == 401:
                    return get_error_response("no_auth")
                elif response.status_code == 404:
                    return create_error_response(
                        "Note not found",
                        "The note may have been deleted. Use search_nodes() to find available notes",
                        "NOT_FOUND"
                    )
                elif response.status_code == 400:
                    try:
                        error_detail = response.json()
                        return create_error_response(
                            f"Bad request: {error_detail.get('detail', 'Unknown error')}",
                            "Check that the note data is valid",
                            "BAD_REQUEST"
                        )
                    except:
                        return create_error_response(
                            "Bad request - invalid note data",
                            "Check that the note data is valid",
                            "BAD_REQUEST"
                        )
                else:
                    return create_error_response(
                        f"API request failed: HTTP {response.status_code}",
                        "Check your network connection and API configuration",
                        "API_ERROR"
                    )

    except httpx.TimeoutException:
        return create_error_response(
            "Request timed out",
            "The FastGTD API is not responding. Check your FASTGTD_API_URL and network connection",
            "TIMEOUT"
        )
    except Exception as e:
        logger.error(f"update_note failed: {str(e)}")
        return create_error_response(
            f"Failed to update note: {str(e)}",
            "Check your network connection and API configuration",
            "EXCEPTION"
        )


async def get_smart_folder_contents(smart_folder_id: str, limit: int = 100, offset: int = 0, auth_token: str = "", current_node_id: str = "") -> dict:
    """Get the contents of a smart folder by evaluating its rules."""
    logger.info(f"get_smart_folder_contents called - smart_folder_id={smart_folder_id}")

    # Get auth token if not provided
    if not auth_token:
        auth_token = await get_auth_token()

    if not auth_token:
        return get_error_response("no_auth")

    if not smart_folder_id:
        return get_error_response("node_id_required")

    # FastGTD API endpoint for getting smart folder contents
    url = f"{FASTGTD_API_URL}/nodes/{smart_folder_id}/contents"

    # Prepare headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_token}"
    }

    # Query parameters for pagination
    params = {
        "limit": min(limit, 500),  # Cap at 500 for performance
        "offset": max(offset, 0)   # Ensure non-negative offset
    }

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.get(
                url,
                headers=headers,
                params=params,
                timeout=HTTP_TIMEOUT
            )

            if response.status_code == 200:
                contents = response.json()

                # Check if contents is None or not a list
                if contents is None:
                    return create_error_response(
                        "API returned None response",
                        "Check that the smart folder exists and has valid rules",
                        "INVALID_RESPONSE"
                    )

                if not isinstance(contents, list):
                    return create_error_response(
                        f"API returned unexpected type: {type(contents)}",
                        "Check that the smart folder exists and has valid rules",
                        "INVALID_RESPONSE"
                    )

                # Extract relevant information from each matching node
                processed_contents = []
                for node in contents:
                    try:
                        node_info = {
                            'id': node.get('id'),
                            'title': node.get('title'),
                            'node_type': node.get('node_type'),
                            'created_at': node.get('created_at'),
                            'updated_at': node.get('updated_at'),
                            'parent_id': node.get('parent_id'),
                            'tags': [tag.get('name', '') for tag in (node.get('tags') or [])]
                        }

                        # Add type-specific data
                        if node.get('node_type') == 'task':
                            task_data = node.get('task_data') or {}
                            description = task_data.get('description') or '' if task_data else ''
                            # Ensure description is string before slicing
                            if not isinstance(description, str):
                                description = str(description) if description else ''
                            node_info.update({
                                'status': task_data.get('status') if task_data else None,
                                'priority': task_data.get('priority') if task_data else None,
                                'due_at': task_data.get('due_at') if task_data else None,
                                'description': description[:100] + ('...' if len(description) > 100 else '')
                            })
                        elif node.get('node_type') == 'note':
                            note_data = node.get('note_data') or {}
                            body = note_data.get('body') or '' if note_data else ''
                            # Ensure body is string before slicing
                            if not isinstance(body, str):
                                body = str(body) if body else ''
                            node_info.update({
                                'content_preview': body[:100] + ('...' if len(body) > 100 else '')
                            })

                        processed_contents.append(node_info)
                    except Exception as e:
                        logger.warning(f"Error processing node {node.get('id', 'unknown')}: {str(e)}")
                        continue  # Skip this node and continue with others

                return {
                    "success": True,
                    "message": f"Found {len(processed_contents)} item(s) matching smart folder rules",
                    "smart_folder_id": smart_folder_id,
                    "contents": processed_contents,
                    "total_shown": len(processed_contents),
                    "pagination": {
                        "limit": limit,
                        "offset": offset,
                        "has_more": len(contents) == limit  # If we got full limit, there might be more
                    }
                }
            else:
                if response.status_code == 401:
                    return get_error_response("no_auth")
                elif response.status_code == 404:
                    return create_error_response(
                        "Smart folder not found",
                        "The smart folder may have been deleted. Use search_nodes() to find available smart folders",
                        "NOT_FOUND"
                    )
                elif response.status_code == 400:
                    try:
                        error_detail = response.json()
                        return create_error_response(
                            f"Bad request: {error_detail.get('detail', 'Invalid smart folder request')}",
                            "Check that the smart folder ID is valid",
                            "BAD_REQUEST"
                        )
                    except:
                        return create_error_response(
                            "Bad request - invalid smart folder ID",
                            "Check that the smart folder ID is valid",
                            "BAD_REQUEST"
                        )
                else:
                    return create_error_response(
                        f"API request failed: HTTP {response.status_code}",
                        "Check your network connection and API configuration",
                        "API_ERROR"
                    )

    except httpx.TimeoutException:
        return create_error_response(
            "Request timed out",
            "The FastGTD API is not responding. Check your FASTGTD_API_URL and network connection",
            "TIMEOUT"
        )
    except Exception as e:
        logger.error(f"get_smart_folder_contents failed: {str(e)}")
        return create_error_response(
            f"Failed to get smart folder contents: {str(e)}",
            "Check your network connection and API configuration",
            "EXCEPTION"
        )


async def instantiate_template(template_id: str, name: str, parent_id: str = "", auth_token: str = "", current_node_id: str = "") -> dict:
    """Create a new instance from a template with all its contents."""
    logger.info(f"instantiate_template called - template_id={template_id}, name={name}")

    # Get auth token if not provided
    if not auth_token:
        auth_token = await get_auth_token()

    if not auth_token:
        return get_error_response("no_auth")

    if not template_id:
        return create_error_response(
            "Template ID is required",
            "Provide a template ID to instantiate. Use list_templates() or search_templates() to find templates",
            "MISSING_TEMPLATE_ID"
        )

    if not name:
        return create_error_response(
            "Instance name is required",
            "Provide a name for the new template instance",
            "MISSING_NAME"
        )

    # FastGTD API endpoint for instantiating templates
    url = f"{FASTGTD_API_URL}/nodes/templates/{template_id}/instantiate"

    # Prepare headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_token}"
    }

    # Build query parameters
    params = {"name": name}

    # Add parent_id if provided
    if parent_id:
        params["parent_id"] = parent_id

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.post(
                url,
                headers=headers,
                params=params,
                timeout=HTTP_TIMEOUT
            )

            if response.status_code == 200:
                created_instance = response.json()

                return {
                    "success": True,
                    "message": f"Template instantiated successfully as '{created_instance.get('title', 'Unknown')}'",
                    "instance": {
                        "id": created_instance["id"],
                        "title": created_instance["title"],
                        "node_type": created_instance["node_type"],
                        "parent_id": created_instance.get("parent_id"),
                        "created_at": created_instance["created_at"],
                        "children_count": created_instance.get("children_count", 0),
                        "is_list": created_instance.get("is_list", False)
                    },
                    "template_id": template_id
                }
            else:
                if response.status_code == 401:
                    return get_error_response("no_auth")
                elif response.status_code == 404:
                    return create_error_response(
                        "Template not found",
                        "The template may have been deleted. Use list_templates() to find available templates",
                        "NOT_FOUND"
                    )
                elif response.status_code == 400:
                    try:
                        error_detail = response.json()
                        return create_error_response(
                            f"Bad request: {error_detail.get('detail', 'Unknown error')}",
                            "Check that the template ID and parameters are valid",
                            "BAD_REQUEST"
                        )
                    except:
                        return create_error_response(
                            "Bad request - invalid template or parameters",
                            "Check that the template ID and parameters are valid",
                            "BAD_REQUEST"
                        )
                else:
                    return create_error_response(
                        f"API request failed: HTTP {response.status_code}",
                        "Check your network connection and API configuration",
                        "API_ERROR"
                    )

    except httpx.TimeoutException:
        return create_error_response(
            "Request timed out",
            "The FastGTD API is not responding. Check your FASTGTD_API_URL and network connection",
            "TIMEOUT"
        )
    except Exception as e:
        logger.error(f"instantiate_template failed: {str(e)}")
        return create_error_response(
            f"Failed to instantiate template: {str(e)}",
            "Check your network connection and API configuration",
            "EXCEPTION"
        )


async def list_templates(category: str = "", limit: int = 50, offset: int = 0, auth_token: str = "", current_node_id: str = "") -> dict:
    """List all available templates with optional category filter."""
    logger.info(f"list_templates called - category={category}")

    # Get auth token if not provided
    if not auth_token:
        auth_token = await get_auth_token()

    if not auth_token:
        return get_error_response("no_auth")

    # FastGTD API endpoint for listing templates
    url = f"{FASTGTD_API_URL}/nodes/templates"

    # Prepare headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_token}"
    }

    # Query parameters
    params = {
        "limit": min(limit, 100),  # Cap at 100 for performance
        "offset": max(offset, 0)   # Ensure non-negative offset
    }

    # Add category filter if provided
    if category:
        params["category"] = category

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.get(
                url,
                headers=headers,
                params=params,
                timeout=HTTP_TIMEOUT
            )

            if response.status_code == 200:
                templates = response.json()

                # Process templates for cleaner output
                processed_templates = []
                for template in templates:
                    template_data = template.get('template_data', {})
                    processed_templates.append({
                        'id': template['id'],
                        'title': template['title'],
                        'description': template_data.get('description', ''),
                        'category': template_data.get('category', ''),
                        'usage_count': template_data.get('usage_count', 0),
                        'created_at': template['created_at'],
                        'updated_at': template['updated_at'],
                        'create_container': template_data.get('create_container', True),
                        'target_node_id': template_data.get('target_node_id')
                    })

                return {
                    "success": True,
                    "message": f"Found {len(processed_templates)} template(s)",
                    "templates": processed_templates,
                    "total_shown": len(processed_templates),
                    "pagination": {
                        "limit": limit,
                        "offset": offset,
                        "has_more": len(templates) == limit  # If we got full limit, there might be more
                    },
                    "category_filter": category if category else "all categories"
                }
            else:
                if response.status_code == 401:
                    return get_error_response("no_auth")
                elif response.status_code == 400:
                    try:
                        error_detail = response.json()
                        return create_error_response(
                            f"Bad request: {error_detail.get('detail', 'Invalid request')}",
                            "Check that the parameters are valid",
                            "BAD_REQUEST"
                        )
                    except:
                        return create_error_response(
                            "Bad request - invalid parameters",
                            "Check that the parameters are valid",
                            "BAD_REQUEST"
                        )
                else:
                    return create_error_response(
                        f"API request failed: HTTP {response.status_code}",
                        "Check your network connection and API configuration",
                        "API_ERROR"
                    )

    except httpx.TimeoutException:
        return create_error_response(
            "Request timed out",
            "The FastGTD API is not responding. Check your FASTGTD_API_URL and network connection",
            "TIMEOUT"
        )
    except Exception as e:
        logger.error(f"list_templates failed: {str(e)}")
        return create_error_response(
            f"Failed to list templates: {str(e)}",
            "Check your network connection and API configuration",
            "EXCEPTION"
        )


async def search_templates(query: str, category: str = "", limit: int = 50, offset: int = 0, auth_token: str = "", current_node_id: str = "") -> dict:
    """Search for templates by name or description."""
    logger.info(f"search_templates called - query={query}, category={category}")

    # Get auth token if not provided
    if not auth_token:
        auth_token = await get_auth_token()

    if not auth_token:
        return get_error_response("no_auth")

    if not query or len(query.strip()) < 1:
        return get_error_response("search_query_required")

    # FastGTD API endpoint for searching nodes (templates)
    url = f"{FASTGTD_API_URL}/nodes/"

    # Prepare headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_token}"
    }

    # Query parameters for search
    params = {
        "search": query.strip(),
        "node_type": "template",
        "limit": min(limit, MAX_PAGE_SIZE),  # Cap at MAX_PAGE_SIZE for performance
        "offset": max(offset, 0)  # Ensure non-negative offset
    }

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.get(
                url,
                headers=headers,
                params=params,
                timeout=HTTP_TIMEOUT
            )

            if response.status_code == 200:
                all_templates = response.json()

                # Filter by category if specified
                filtered_templates = []
                for template in all_templates:
                    template_data = template.get('template_data', {})
                    template_category = template_data.get('category', '')

                    # Apply category filter
                    if category and template_category.lower() != category.lower():
                        continue

                    filtered_templates.append({
                        'id': template['id'],
                        'title': template['title'],
                        'description': template_data.get('description', ''),
                        'category': template_category,
                        'usage_count': template_data.get('usage_count', 0),
                        'created_at': template['created_at'],
                        'updated_at': template['updated_at'],
                        'create_container': template_data.get('create_container', True),
                        'target_node_id': template_data.get('target_node_id')
                    })

                return {
                    "success": True,
                    "message": f"Found {len(filtered_templates)} template(s) matching '{query}'",
                    "query": query,
                    "category_filter": category if category else "all categories",
                    "templates": filtered_templates,
                    "total_found": len(filtered_templates)
                }
            else:
                if response.status_code == 401:
                    return get_error_response("no_auth")
                elif response.status_code == 400:
                    try:
                        error_detail = response.json()
                        return create_error_response(
                            f"Bad request: {error_detail.get('detail', 'Invalid search query')}",
                            "Check that the search parameters are valid",
                            "BAD_REQUEST"
                        )
                    except:
                        return create_error_response(
                            "Bad request - invalid search parameters",
                            "Check that the search parameters are valid",
                            "BAD_REQUEST"
                        )
                else:
                    return create_error_response(
                        f"API request failed: HTTP {response.status_code}",
                        "Check your network connection and API configuration",
                        "API_ERROR"
                    )

    except httpx.TimeoutException:
        return create_error_response(
            "Request timed out",
            "The FastGTD API is not responding. Check your FASTGTD_API_URL and network connection",
            "TIMEOUT"
        )
    except Exception as e:
        logger.error(f"search_templates failed: {str(e)}")
        return create_error_response(
            f"Failed to search templates: {str(e)}",
            "Check your network connection and API configuration",
            "EXCEPTION"
        )

async def upload_artifact(node_id: str, file_path: str, filename: str = "", auth_token: str = "", current_node_id: str = "") -> dict:
    """Upload a file and attach it to a node."""
    logger.info(f"upload_artifact called - node_id={node_id}, file_path={file_path}")

    # Get auth token if not provided
    if not auth_token:
        auth_token = await get_auth_token()

    if not auth_token:
        return get_error_response("no_auth")

    if not node_id:
        return get_error_response("node_id_required")

    if not file_path:
        return create_error_response(
            "file_path is required",
            "Provide the path to the file you want to upload",
            "MISSING_FILE_PATH"
        )

    # Check if file exists
    from pathlib import Path
    try:
        file_pathobj = Path(file_path)
        if not file_pathobj.exists():
            return create_error_response(
                f"File not found: {file_path}",
                "Check that the file path is correct and the file exists",
                "FILE_NOT_FOUND"
            )
    except Exception as e:
        return create_error_response(
            f"Invalid file path: {str(e)}",
            "Provide a valid file path",
            "INVALID_PATH"
        )

    # Use custom filename or default to file basename
    upload_filename = filename if filename else file_pathobj.name

    # FastGTD API endpoint for uploading artifacts
    url = f"{FASTGTD_API_URL}/artifacts"

    # Prepare headers
    headers = {
        "Authorization": f"Bearer {auth_token}"
    }

    try:
        # Prepare multipart form data
        with open(file_path, "rb") as f:
            files = {"file": (upload_filename, f, None)}
            data = {"node_id": node_id}

            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
                response = await client.post(
                    url,
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=HTTP_TIMEOUT
                )

        if response.status_code == 201:
            artifact_data = response.json()
            logger.info(f"Successfully uploaded artifact {artifact_data['id']} to node {node_id}")
            return {
                "success": True,
                "message": f"Successfully uploaded '{upload_filename}' to node {node_id}",
                "artifact": artifact_data
            }
        else:
            if response.status_code == 401:
                return get_error_response("no_auth")
            elif response.status_code == 404:
                return create_error_response(
                    f"Node {node_id} not found",
                    "The node may have been deleted or you don't have access. Use search_nodes() to find available nodes",
                    "NOT_FOUND"
                )
            elif response.status_code == 400:
                try:
                    error_detail = response.json()
                    return create_error_response(
                        f"Bad request: {error_detail.get('detail', 'Invalid upload parameters')}",
                        "Check that the node ID and file are valid",
                        "BAD_REQUEST"
                    )
                except:
                    return create_error_response(
                        "Bad request - invalid upload parameters",
                        "Check that the node ID and file are valid",
                        "BAD_REQUEST"
                    )
            else:
                return create_error_response(
                    f"Upload failed: HTTP {response.status_code}",
                    "Check your network connection and API configuration",
                    "API_ERROR"
                )

    except IOError as e:
        return create_error_response(
            f"File read error: {str(e)}",
            "Check that the file is readable and not locked by another process",
            "FILE_READ_ERROR"
        )
    except httpx.TimeoutException:
        return create_error_response(
            "Request timed out",
            "The FastGTD API is not responding. Check your FASTGTD_API_URL and network connection",
            "TIMEOUT"
        )
    except Exception as e:
        logger.error(f"upload_artifact failed: {str(e)}")
        return create_error_response(
            f"Failed to upload artifact: {str(e)}",
            "Check your network connection and API configuration",
            "EXCEPTION"
        )

async def download_artifact(artifact_id: str, download_path: str = "", auth_token: str = "", current_node_id: str = "") -> dict:
    """Download an artifact file by ID."""
    logger.info(f"download_artifact called - artifact_id={artifact_id}, download_path={download_path}")

    # Get auth token if not provided
    if not auth_token:
        auth_token = await get_auth_token()

    if not auth_token:
        return get_error_response("no_auth")

    if not artifact_id:
        return create_error_response(
            "artifact_id is required",
            "Provide the ID of the artifact you want to download",
            "MISSING_ARTIFACT_ID"
        )

    # FastGTD API endpoint for downloading artifacts
    url = f"{FASTGTD_API_URL}/artifacts/{artifact_id}/download"

    # Prepare headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_token}"
    }

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.get(url, headers=headers, timeout=HTTP_TIMEOUT)

        if response.status_code == 200:
            # Get filename from Content-Disposition header or use artifact_id as fallback
            import re
            from pathlib import Path
            filename = artifact_id
            if 'content-disposition' in response.headers:
                disposition = response.headers['content-disposition']
                filename_match = re.search(r'filename="?([^"]+)"?', disposition)
                if filename_match:
                    filename = filename_match.group(1)

            # Determine save path with better defaults for LLM environments
            if download_path:
                save_path = Path(download_path)
                if save_path.is_dir():
                    save_path = save_path / filename
            else:
                # Use configurable default download path
                save_path = Path(DEFAULT_DOWNLOAD_PATH) / filename

            # Try to write file content with fallback handling
            try:
                with open(save_path, "wb") as f:
                    f.write(response.content)

                logger.info(f"Successfully downloaded artifact {artifact_id} to {save_path}")
                return {
                    "success": True,
                    "message": f"Successfully downloaded artifact to '{save_path}'",
                    "file_path": str(save_path),
                    "size_bytes": len(response.content)
                }
            except (OSError, PermissionError) as e:
                # If writing fails (read-only filesystem), return content instead
                logger.warning(f"Could not write file {save_path}: {e}, returning content instead")
                try:
                    # Try to decode as text for better display
                    content_text = response.content.decode('utf-8')
                    logger.info(f"Downloaded artifact {artifact_id} content as text (filesystem readonly)")
                    return {
                        "success": True,
                        "message": f"Downloaded artifact content (could not write to file: {e})",
                        "file_path": "content_returned_due_to_readonly_filesystem",
                        "size_bytes": len(response.content),
                        "content": content_text,
                        "warning": f"File system write failed: {e}"
                    }
                except UnicodeDecodeError:
                    # Binary content - return as base64
                    import base64
                    content_b64 = base64.b64encode(response.content).decode('ascii')
                    logger.info(f"Downloaded artifact {artifact_id} content as base64 (filesystem readonly)")
                    return {
                        "success": True,
                        "message": f"Downloaded binary artifact content (could not write to file: {e})",
                        "file_path": "binary_content_returned_due_to_readonly_filesystem",
                        "size_bytes": len(response.content),
                        "content_base64": content_b64,
                        "warning": f"File system write failed: {e}"
                    }
        else:
            if response.status_code == 401:
                return get_error_response("no_auth")
            elif response.status_code == 404:
                return create_error_response(
                    f"Artifact {artifact_id} not found",
                    "The artifact may have been deleted or you don't have access",
                    "NOT_FOUND"
                )
            else:
                return create_error_response(
                    f"Download failed: HTTP {response.status_code}",
                    "Check your network connection and API configuration",
                    "API_ERROR"
                )

    except httpx.TimeoutException:
        return create_error_response(
            "Request timed out",
            "The FastGTD API is not responding. Check your FASTGTD_API_URL and network connection",
            "TIMEOUT"
        )
    except Exception as e:
        logger.error(f"download_artifact failed: {str(e)}")
        return create_error_response(
            f"Failed to download artifact: {str(e)}",
            "Check your network connection and API configuration",
            "EXCEPTION"
        )

async def delete_artifact(artifact_id: str, auth_token: str = "", current_node_id: str = "") -> dict:
    """Delete an artifact and its associated file."""
    logger.info(f"delete_artifact called - artifact_id={artifact_id}")

    # Get auth token if not provided
    if not auth_token:
        auth_token = await get_auth_token()

    if not auth_token:
        return get_error_response("no_auth")

    if not artifact_id:
        return create_error_response(
            "artifact_id is required",
            "Provide the ID of the artifact you want to delete",
            "MISSING_ARTIFACT_ID"
        )

    # FastGTD API endpoint for deleting artifacts
    url = f"{FASTGTD_API_URL}/artifacts/{artifact_id}"

    # Prepare headers
    headers = {
        "Authorization": f"Bearer {auth_token}"
    }

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.delete(url, headers=headers, timeout=HTTP_TIMEOUT)

        if response.status_code == 204:
            logger.info(f"Successfully deleted artifact {artifact_id}")
            return {
                "success": True,
                "message": f"Successfully deleted artifact {artifact_id}"
            }
        else:
            if response.status_code == 401:
                return get_error_response("no_auth")
            elif response.status_code == 404:
                return create_error_response(
                    f"Artifact {artifact_id} not found",
                    "The artifact may have been deleted or you don't have access",
                    "NOT_FOUND"
                )
            else:
                return create_error_response(
                    f"Delete failed: HTTP {response.status_code}",
                    "Check your network connection and API configuration",
                    "API_ERROR"
                )

    except httpx.TimeoutException:
        return create_error_response(
            "Request timed out",
            "The FastGTD API is not responding. Check your FASTGTD_API_URL and network connection",
            "TIMEOUT"
        )
    except Exception as e:
        logger.error(f"delete_artifact failed: {str(e)}")
        return create_error_response(
            f"Failed to delete artifact: {str(e)}",
            "Check your network connection and API configuration",
            "EXCEPTION"
        )

async def list_node_artifacts(node_id: str, limit: int = DEFAULT_PAGE_SIZE, offset: int = 0, auth_token: str = "", current_node_id: str = "") -> dict:
    """List all artifacts attached to a specific node."""
    logger.info(f"list_node_artifacts called - node_id={node_id}")

    # Get auth token if not provided
    if not auth_token:
        auth_token = await get_auth_token()

    if not auth_token:
        return get_error_response("no_auth")

    if not node_id:
        return get_error_response("node_id_required")

    # FastGTD API endpoint for listing node artifacts
    url = f"{FASTGTD_API_URL}/artifacts/node/{node_id}"

    # Prepare headers
    headers = {
        "Authorization": f"Bearer {auth_token}"
    }

    # Prepare pagination parameters
    params = {
        "limit": min(limit, MAX_PAGE_SIZE),
        "offset": max(offset, 0)
    }

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.get(url, headers=headers, params=params, timeout=HTTP_TIMEOUT)

        if response.status_code == 200:
            artifacts_data = response.json()
            logger.info(f"Found {artifacts_data.get('total', 0)} artifacts for node {node_id}")
            return {
                "success": True,
                "message": f"Found {artifacts_data.get('total', 0)} artifact(s) for node {node_id}",
                "artifacts": artifacts_data.get('artifacts', []),
                "total": artifacts_data.get('total', 0)
            }
        else:
            if response.status_code == 401:
                return get_error_response("no_auth")
            elif response.status_code == 404:
                return create_error_response(
                    f"Node {node_id} not found",
                    "The node may have been deleted or you don't have access. Use search_nodes() to find available nodes",
                    "NOT_FOUND"
                )
            else:
                return create_error_response(
                    f"List failed: HTTP {response.status_code}",
                    "Check your network connection and API configuration",
                    "API_ERROR"
                )

    except httpx.TimeoutException:
        return create_error_response(
            "Request timed out",
            "The FastGTD API is not responding. Check your FASTGTD_API_URL and network connection",
            "TIMEOUT"
        )
    except Exception as e:
        logger.error(f"list_node_artifacts failed: {str(e)}")
        return create_error_response(
            f"Failed to list node artifacts: {str(e)}",
            "Check your network connection and API configuration",
            "EXCEPTION"
        )

# Tool handlers mapping
TOOL_HANDLERS = {
    "add_task_to_inbox": add_task_to_inbox,
    "add_task_to_current_node": add_task_to_current_node,
    "add_folder_to_current_node": add_folder_to_current_node,
    "add_note_to_current_node": add_note_to_current_node,
    "add_note_to_node_id": add_note_to_node_id,
    "get_all_folders": get_all_folders,
    "get_root_folders": get_root_folders,
    "get_root_nodes": get_root_nodes,
    "get_node_children": get_node_children,
    "get_folder_id": get_folder_id,
    "add_task_to_node_id": add_task_to_node_id,
    "get_node_tree": get_node_tree,
    "search_nodes": search_nodes,
    "create_task": create_task,
    "update_task": update_task,
    "complete_task": complete_task,
    "delete_task": delete_task,
    "delete_folder": delete_folder,
    "create_folder": create_folder,
    "move_node": move_node,
    "add_tag": add_tag,
    "remove_tag": remove_tag,
    "get_today_tasks": get_today_tasks,
    "get_overdue_tasks": get_overdue_tasks,
    "update_note": update_note,
    "get_smart_folder_contents": get_smart_folder_contents,
    "instantiate_template": instantiate_template,
    "list_templates": list_templates,
    "search_templates": search_templates,
    "upload_artifact": upload_artifact,
    "download_artifact": download_artifact,
    "delete_artifact": delete_artifact,
    "list_node_artifacts": list_node_artifacts,
}

@server.list_tools()
async def handle_list_tools():
    """List available FastGTD tools."""
    tools = [
        Tool(
            name="add_task_to_inbox",
            description="Add a new task to user's inbox (default node) - perfect for quick task capture",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Task title (required)"},
                    "description": {"type": "string", "description": "Task description (optional)"},
                    "priority": {"type": "string", "description": "Priority: low, medium, high, urgent (default: medium)"}
                },
                "required": ["title"]
            },
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=False,
            openWorldHint=True
        ),
        Tool(
            name="add_task_to_current_node",
            description="Add a new task to the user's currently selected node/folder",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Task title (required)"},
                    "description": {"type": "string", "description": "Task description (optional)"},
                    "priority": {"type": "string", "description": "Priority: low, medium, high, urgent (default: medium)"}
                },
                "required": ["title"]
            },
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=False,
            openWorldHint=True
        ),
        Tool(
            name="add_folder_to_current_node",
            description="Add a new folder to the user's currently selected node/folder",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Folder name (required)"},
                    "description": {"type": "string", "description": "Optional description for the folder"}
                },
                "required": ["title"]
            },
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=False,
            openWorldHint=True
        ),
        Tool(
            name="add_note_to_current_node",
            description="Add a new note to the user's currently selected node/folder",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Note title (required)"},
                    "content": {"type": "string", "description": "Note content/body (optional)"}
                },
                "required": ["title"]
            },
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=False,
            openWorldHint=True
        ),
        Tool(
            name="add_note_to_node_id",
            description="Add a new note to a specific node by its ID - allows programmatic note creation without relying on current node context",
            inputSchema={
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "ID of the parent node to add the note to (required)"},
                    "title": {"type": "string", "description": "Note title (required)"},
                    "content": {"type": "string", "description": "Note content/body (optional)"}
                },
                "required": ["node_id", "title"]
            },
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=False,
            openWorldHint=True
        ),
        Tool(
            name="get_all_folders",
            description="Get all folder names in the user's node tree - useful for AI to help find the right folder when user mentions one",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Maximum number of folders to return (default: 50, max: 100)"},
                    "offset": {"type": "integer", "description": "Number of folders to skip for pagination (default: 0)"}
                },
                "required": []
            },
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True
        ),
        Tool(
            name="get_root_folders",
            description="Get only root-level folders (folders with no parent) - useful for showing top-level organization",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Maximum number of folders to return (default: 50, max: 100)"},
                    "offset": {"type": "integer", "description": "Number of folders to skip for pagination (default: 0)"}
                },
                "required": []
            },
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True
        ),
        Tool(
            name="get_root_nodes",
            description="Get all root-level nodes (all types with no parent) - tasks, notes, folders, smart folders, templates - complete root overview",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Maximum number of nodes to return (default: 50, max: 100)"},
                    "offset": {"type": "integer", "description": "Number of nodes to skip for pagination (default: 0)"}
                },
                "required": []
            },
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True
        ),
        Tool(
            name="get_node_children",
            description="Get immediate children of a specific node (optionally filtered by type) - perfect for browsing folder contents or exploring hierarchy",
            inputSchema={
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "ID of the parent node to get children from (required)"},
                    "node_type": {"type": "string", "description": "Filter by node type: task, note, folder, smart_folder, template (optional)"},
                    "limit": {"type": "integer", "description": "Maximum number of children to return (default: 50, max: 100)"},
                    "offset": {"type": "integer", "description": "Number of children to skip for pagination (default: 0)"}
                },
                "required": ["node_id"]
            },
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True
        ),
        Tool(
            name="get_folder_id",
            description="Get folder ID by folder name - useful for finding the specific folder to work with",
            inputSchema={
                "type": "object",
                "properties": {
                    "folder_name": {"type": "string", "description": "Name of the folder to find (required)"}
                },
                "required": ["folder_name"]
            },
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True
        ),
        Tool(
            name="add_task_to_node_id",
            description="Add a task to a specific node by node ID and return the new task's ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "ID of the node to add the task to (required)"},
                    "task_title": {"type": "string", "description": "Title of the task to create (required)"},
                    "description": {"type": "string", "description": "Task description (optional)"},
                    "priority": {"type": "string", "description": "Priority: low, medium, high, urgent (default: medium)"}
                },
                "required": ["node_id", "task_title"]
            },
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=False,
            openWorldHint=True
        ),
        Tool(
            name="get_node_tree",
            description="Get the hierarchical node tree structure - perfect for browsing and exploring the user's organization",
            inputSchema={
                "type": "object",
                "properties": {
                    "root_id": {"type": "string", "description": "ID of the root node to start from (optional, defaults to user's root)"},
                    "max_depth": {"type": "integer", "description": "Maximum depth to traverse (default: 10, max: 20)"}
                },
                "required": []
            },
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True
        ),
        Tool(
            name="create_task",
            description="Create a new task with simplified interface - automatically finds best location (inbox/current folder)",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Task title (required)"},
                    "description": {"type": "string", "description": "Task description (optional)"},
                    "priority": {"type": "string", "description": "Priority: low, medium, high, urgent (default: medium)"},
                    "parent_id": {"type": "string", "description": "Specific parent folder ID (optional - auto-detects if not provided)"}
                },
                "required": ["title"]
            },
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=False,
            openWorldHint=True
        ),
        Tool(
            name="update_task",
            description="Update an existing task's properties - title, description, priority, status, dates, etc.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "ID of the task to update (required)"},
                    "title": {"type": "string", "description": "New task title (optional)"},
                    "description": {"type": "string", "description": "New task description (optional)"},
                    "priority": {"type": "string", "description": "New priority: low, medium, high, urgent (optional)"},
                    "status": {"type": "string", "description": "New status: todo, in_progress, done, dropped (optional)"},
                    "due_at": {"type": "string", "description": "Due date/time in ISO format (optional, e.g., '2024-12-25T10:00:00Z')"},
                    "earliest_start_at": {"type": "string", "description": "Earliest start date/time in ISO format (optional)"},
                    "archived": {"type": "boolean", "description": "Whether task is archived (optional)"},
                    "recurrence_rule": {"type": "string", "description": "Recurrence rule for recurring tasks (optional)"},
                    "recurrence_anchor": {"type": "string", "description": "Recurrence anchor date in ISO format (optional)"}
                },
                "required": ["task_id"]
            },
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True
        ),
        Tool(
            name="search_nodes",
            description="Search for nodes by title and content - perfect for finding specific tasks, notes, folders, or any content",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search term to look for in titles and content (required)"},
                    "node_type": {"type": "string", "description": "Filter by node type: 'task', 'note', 'folder', 'smart_folder', or 'template' (optional)"},
                    "limit": {"type": "integer", "description": "Maximum number of results to return (default: 50, max: 100)"},
                    "offset": {"type": "integer", "description": "Number of items to skip for pagination (default: 0)"}
                },
                "required": ["query"]
            },
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True
        ),
        Tool(
            name="complete_task",
            description="Mark a task as completed",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "ID of the task to complete (required)"}
                },
                "required": ["task_id"]
            },
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True
        ),
        Tool(
            name="delete_task",
            description="Delete a task permanently",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "ID of the task to delete (required)"}
                },
                "required": ["task_id"]
            },
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=True,
            openWorldHint=True
        ),
        Tool(
            name="delete_folder",
            description="Delete a folder permanently (will also delete all contents inside the folder)",
            inputSchema={
                "type": "object",
                "properties": {
                    "folder_id": {"type": "string", "description": "ID of the folder to delete (required)"}
                },
                "required": ["folder_id"]
            },
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=True,
            openWorldHint=True
        ),
        Tool(
            name="create_folder",
            description="Create a new folder - automatically finds best location (current folder or root)",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Folder title/name (required)"},
                    "description": {"type": "string", "description": "Optional description for the folder"},
                    "parent_id": {"type": "string", "description": "Specific parent folder ID (optional - auto-detects if not provided)"}
                },
                "required": ["title"]
            },
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=False,
            openWorldHint=True
        ),
        Tool(
            name="move_node",
            description="Move a task, note, or folder to a different location in the hierarchy",
            inputSchema={
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "ID of the node to move (required)"},
                    "new_parent_id": {"type": "string", "description": "ID of the new parent folder (optional - leave empty to move to root)"},
                    "new_sort_order": {"type": "integer", "description": "New position/order within the parent (optional)"}
                },
                "required": ["node_id"]
            },
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True
        ),
        Tool(
            name="add_tag",
            description="Add a tag to a task, note, or folder (creates tag if it doesn't exist)",
            inputSchema={
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "ID of the node to tag (required)"},
                    "tag_name": {"type": "string", "description": "Name of the tag (required)"},
                    "tag_description": {"type": "string", "description": "Optional description for the tag"},
                    "tag_color": {"type": "string", "description": "Optional hex color code for the tag (e.g., #FF0000)"}
                },
                "required": ["node_id", "tag_name"]
            },
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True
        ),
        Tool(
            name="remove_tag",
            description="Remove a tag from a task, note, or folder",
            inputSchema={
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "ID of the node to remove tag from (required)"},
                    "tag_name": {"type": "string", "description": "Name of the tag to remove (required)"}
                },
                "required": ["node_id", "tag_name"]
            },
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True
        ),
        Tool(
            name="get_today_tasks",
            description="Get all tasks that are due today",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            },
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True
        ),
        Tool(
            name="get_overdue_tasks",
            description="Get all tasks that are overdue (due date in the past)",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            },
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True
        ),
        Tool(
            name="update_note",
            description="Update an existing note's title and/or content",
            inputSchema={
                "type": "object",
                "properties": {
                    "note_id": {"type": "string", "description": "ID of the note to update (required)"},
                    "title": {"type": "string", "description": "New title for the note (optional)"},
                    "content": {"type": "string", "description": "New content/body for the note (optional)"}
                },
                "required": ["note_id"]
            },
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True
        ),
        Tool(
            name="get_smart_folder_contents",
            description="Get the contents of a smart folder by evaluating its rules",
            inputSchema={
                "type": "object",
                "properties": {
                    "smart_folder_id": {"type": "string", "description": "ID of the smart folder (required)"},
                    "limit": {"type": "integer", "description": "Maximum number of items to return (default: 100, max: 500)"},
                    "offset": {"type": "integer", "description": "Number of items to skip for pagination (default: 0)"}
                },
                "required": ["smart_folder_id"]
            },
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True
        ),
        Tool(
            name="instantiate_template",
            description="Create a new instance from a template with all its contents",
            inputSchema={
                "type": "object",
                "properties": {
                    "template_id": {"type": "string", "description": "ID of the template to instantiate (required)"},
                    "name": {"type": "string", "description": "Name for the new instance (required)"},
                    "parent_id": {"type": "string", "description": "Parent folder ID where to create the instance (optional)"}
                },
                "required": ["template_id", "name"]
            },
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=False,
            openWorldHint=True
        ),
        Tool(
            name="list_templates",
            description="List all available templates with optional category filter",
            inputSchema={
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "Filter by category (optional)"},
                    "limit": {"type": "integer", "description": "Maximum number of templates to return (default: 50, max: 100)"},
                    "offset": {"type": "integer", "description": "Number of templates to skip for pagination (default: 0)"}
                },
                "required": []
            },
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True
        ),
        Tool(
            name="search_templates",
            description="Search for templates by name or description",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query to match template names or descriptions (required)"},
                    "category": {"type": "string", "description": "Filter by category (optional)"},
                    "limit": {"type": "integer", "description": "Maximum number of templates to return (default: 50, max: 100)"},
                    "offset": {"type": "integer", "description": "Number of templates to skip for pagination (default: 0)"}
                },
                "required": ["query"]
            },
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True
        ),
        Tool(
            name="upload_artifact",
            description="Upload a file and attach it to a node",
            inputSchema={
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "UUID of the node to attach the file to (required)"},
                    "file_path": {"type": "string", "description": "Path to the file to upload (required)"},
                    "filename": {"type": "string", "description": "Optional filename to use for the upload (defaults to basename of file_path)"}
                },
                "required": ["node_id", "file_path"]
            },
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=False,
            openWorldHint=True
        ),
        Tool(
            name="download_artifact",
            description="Download an artifact file by ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "artifact_id": {"type": "string", "description": "UUID of the artifact to download (required)"},
                    "download_path": {"type": "string", "description": "Local path where to save the downloaded file (optional, defaults to original filename)"}
                },
                "required": ["artifact_id"]
            },
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True
        ),
        Tool(
            name="delete_artifact",
            description="Delete an artifact and its associated file",
            inputSchema={
                "type": "object",
                "properties": {
                    "artifact_id": {"type": "string", "description": "UUID of the artifact to delete (required)"}
                },
                "required": ["artifact_id"]
            },
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=True,
            openWorldHint=True
        ),
        Tool(
            name="list_node_artifacts",
            description="List all artifacts attached to a specific node",
            inputSchema={
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "UUID of the node to get artifacts for (required)"},
                    "limit": {"type": "integer", "description": "Maximum number of artifacts to return (default: 50, max: 100)"},
                    "offset": {"type": "integer", "description": "Number of artifacts to skip for pagination (default: 0)"}
                },
                "required": ["node_id"]
            },
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True
        )
    ]
    return tools

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict | None = None):
    """Handle tool execution."""
    
    if not arguments:
        arguments = {}
        
    handler = TOOL_HANDLERS.get(name)
    if handler:
        try:
            result = await handler(**arguments)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            error_info = {"success": False, "error": f"Tool execution failed: {str(e)}"}
            return [TextContent(type="text", text=json.dumps(error_info, indent=2))]
    else:
        raise ValueError(f"Unknown tool: {name}")

async def main():
    """Run the FastGTD test MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="fastgtd-test",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())