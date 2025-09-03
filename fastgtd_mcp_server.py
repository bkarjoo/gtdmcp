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

from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    TextContent,
    Tool,
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
    logger.info(f"🧪 add_task_to_inbox CALLED - title='{title}', description='{description}', priority='{priority}', auth_token_present={bool(auth_token)}")
    
    try:
        import httpx
        
        # Get auth token if not provided
        if not auth_token:
            auth_token = await get_auth_token()
        
        print(f"🧪 MCP DEBUG - add_task_to_inbox called:")
        print(f"   Title: {title}")
        print(f"   Description: {description}")
        print(f"   Priority: {priority}")
        print(f"   Auth token present: {bool(auth_token)}")
        
        # FastGTD API endpoint
        url = f"{FASTGTD_API_URL}/nodes/"
    
    except Exception as e:
        print(f"❌ MCP ERROR in setup: {str(e)}")
        return {
            "success": False,
            "error": f"MCP tool setup failed: {str(e)}"
        }
    
    # Create task payload for unified node system
    task_payload = {
        "node_type": "task",
        "title": title,
        "task_data": {
            "description": description,
            "priority": priority,
            "status": "todo",
            "archived": False
        }
    }
    
    # Prepare headers
    headers = {"Content-Type": "application/json"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    
    try:
        async with httpx.AsyncClient() as client:
            # First, get the user's default node (inbox)
            if auth_token:
                default_node_response = await client.get(
                    f"{FASTGTD_API_URL}/settings/default-node",
                    headers=headers
                )
                
                if default_node_response.status_code == 200:
                    default_data = default_node_response.json()
                    default_node_id = default_data.get("node_id")
                    print(f"📋 Default node response: {default_data}")
                    if default_node_id:
                        task_payload["parent_id"] = default_node_id
                        print(f"🎯 Setting parent_id to default node: {default_node_id}")
                    else:
                        print(f"⚠️  No default node set for user - task will be added to root")
                else:
                    print(f"⚠️  Failed to get default node: HTTP {default_node_response.status_code}")
                    
            print(f"📤 Final task payload: {task_payload}")
            
            response = await client.post(
                url,
                json=task_payload,
                headers=headers
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
                return {
                    "success": False,
                    "error": f"Failed to add task: HTTP {response.status_code}",
                    "details": response.text
                }
                
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to add task to inbox: {str(e)}"
        }

async def add_folder_to_current_node(title: str, auth_token: str = "", current_node_id: str = "") -> dict:
    """Add a folder to the current node"""
    try:
        import httpx
        
        print(f"📁 MCP DEBUG - add_folder_to_current_node called:")
        print(f"   Title: {title}")
        print(f"   Current node ID: {current_node_id}")
        print(f"   Auth token present: {bool(auth_token)}")
        
        if not current_node_id:
            return {"success": False, "error": "No current node ID provided"}
        
        # FastGTD API endpoint
        url = f"{FASTGTD_API_URL}/nodes/"
        
        # Create folder payload - folders are their own node type
        folder_payload = {
            "node_type": "folder",
            "title": title,
            "parent_id": current_node_id
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {auth_token}"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=folder_payload, headers=headers)
            
        if response.status_code == 200:
            folder_data = response.json()
            return {
                "success": True,
                "message": f"Folder '{title}' added to current node successfully",
                "folder_id": folder_data.get("id"),
                "folder": folder_data
            }
        else:
            error_text = response.text
            print(f"❌ API Error {response.status_code}: {error_text}")
            return {
                "success": False,
                "error": f"API request failed with status {response.status_code}: {error_text}"
            }
            
    except Exception as e:
        print(f"❌ MCP ERROR in add_folder_to_current_node: {str(e)}")
        return {
            "success": False,
            "error": f"Tool execution failed: {str(e)}"
        }

async def add_task_to_current_node(title: str, description: str = "", priority: str = "medium", auth_token: str = "", current_node_id: str = "") -> dict:
    """Add a task to the user's currently selected node"""
    try:
        import httpx
        
        # Get auth token if not provided
        if not auth_token:
            auth_token = await get_auth_token()
        
        print(f"🧪 MCP DEBUG - add_task_to_current_node called:")
        print(f"   Title: {title}")
        print(f"   Description: {description}")
        print(f"   Priority: {priority}")
        print(f"   Auth token present: {bool(auth_token)}")
        print(f"   Current node ID: {current_node_id}")
        
        if not current_node_id:
            return {
                "success": False,
                "error": "No current node ID provided - cannot determine where to add task"
            }
        
        # FastGTD API endpoint
        url = f"{FASTGTD_API_URL}/nodes/"
    
    except Exception as e:
        print(f"❌ MCP ERROR in setup: {str(e)}")
        return {
            "success": False,
            "error": f"MCP tool setup failed: {str(e)}"
        }
    
    # Create task payload for unified node system
    task_payload = {
        "node_type": "task",
        "title": title,
        "parent_id": current_node_id,  # Use provided current node directly
        "task_data": {
            "description": description,
            "priority": priority,
            "status": "todo",
            "archived": False
        }
    }
    
    # Prepare headers
    headers = {"Content-Type": "application/json"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    
    print(f"📤 Final task payload: {task_payload}")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json=task_payload,
                headers=headers
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
                return {
                    "success": False,
                    "error": f"Failed to add task: HTTP {response.status_code}",
                    "details": response.text
                }
                
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to add task to current node: {str(e)}"
        }

async def add_note_to_current_node(title: str, content: str = "", auth_token: str = "", current_node_id: str = "") -> dict:
    """Add a note to the user's currently selected node"""
    try:
        import httpx
        
        # Get auth token if not provided
        if not auth_token:
            auth_token = await get_auth_token()
        
        print(f"📝 MCP DEBUG - add_note_to_current_node called:")
        print(f"   Title: {title}")
        print(f"   Content: {content}")
        print(f"   Auth token present: {bool(auth_token)}")
        print(f"   Current node ID: {current_node_id}")
        
        if not current_node_id:
            return {
                "success": False,
                "error": "No current node ID provided - cannot determine where to add note"
            }
        
        # FastGTD API endpoint
        url = f"{FASTGTD_API_URL}/nodes/"
    
    except Exception as e:
        print(f"❌ MCP ERROR in setup: {str(e)}")
        return {
            "success": False,
            "error": f"MCP tool setup failed: {str(e)}"
        }
    
    # Create note payload for unified node system
    note_payload = {
        "node_type": "note",
        "title": title,
        "parent_id": current_node_id,  # Use provided current node directly
        "note_data": {
            "body": content
        }
    }
    
    # Prepare headers
    headers = {"Content-Type": "application/json"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    
    print(f"📤 Final note payload: {note_payload}")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json=note_payload,
                headers=headers
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
                return {
                    "success": False,
                    "error": f"Failed to add note: HTTP {response.status_code}",
                    "details": response.text
                }
                
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to add note to current node: {str(e)}"
        }

async def get_all_folders(auth_token: str = "", current_node_id: str = "") -> dict:
    """Get all folder names in the user's node tree for AI to help find the right folder"""
    try:
        import httpx
        
        # Get auth token if not provided
        if not auth_token:
            auth_token = await get_auth_token()
        
        print(f"📁 MCP DEBUG - get_all_folders called:")
        print(f"   Auth token present: {bool(auth_token)}")
        
        if not auth_token:
            return {"success": False, "error": "No authentication token provided"}
        
        # FastGTD API endpoint - get all notes (folders are notes with "Container folder" body)
        url = f"{FASTGTD_API_URL}/nodes/"
    
    except Exception as e:
        print(f"❌ MCP ERROR in setup: {str(e)}")
        return {
            "success": False,
            "error": f"MCP tool setup failed: {str(e)}"
        }
    
    # Prepare headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_token}"
    }
    
    try:
        async with httpx.AsyncClient() as client:
            # Get all folders directly
            response = await client.get(
                url,
                headers=headers,
                params={"node_type": "folder", "limit": 1000}  # Get all folders
            )
            
            if response.status_code in [200, 201]:
                nodes_data = response.json()
                folders = []
                
                # Extract folder titles
                for node in nodes_data:
                    if node.get("node_type") == "folder":
                        folder_title = node.get("title")
                        if folder_title:
                            folders.append(folder_title)
                
                return {
                    "success": True,
                    "message": f"Found {len(folders)} folders",
                    "folders": folders
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to retrieve folders: HTTP {response.status_code}",
                    "details": response.text
                }
                
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to get folders: {str(e)}"
        }

async def get_root_folders(auth_token: str = "", current_node_id: str = "") -> dict:
    """Get only root-level folders (folders with no parent)"""
    try:
        import httpx
        
        # Get auth token if not provided
        if not auth_token:
            auth_token = await get_auth_token()
        
        print(f"📁 MCP DEBUG - get_root_folders called:")
        print(f"   Auth token present: {bool(auth_token)}")
        
        if not auth_token:
            return {"success": False, "error": "No authentication token provided"}
        
        # FastGTD API endpoint - get folders with no parent (root level)
        url = f"{FASTGTD_API_URL}/nodes/"
    
    except Exception as e:
        print(f"❌ MCP ERROR in setup: {str(e)}")
        return {
            "success": False,
            "error": f"MCP tool setup failed: {str(e)}"
        }
    
    # Prepare headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_token}"
    }
    
    try:
        async with httpx.AsyncClient() as client:
            # Get folders with no parent (root level)
            # Note: We need to explicitly request parent_id=None via query parameter
            # But HTTP doesn't have a clean way to send null, so we'll use a special approach
            
            # First, let's try using an empty string or special value that the API recognizes
            response = await client.get(
                url,
                headers=headers,
                params={"node_type": "folder", "limit": 1000}
            )
            
            if response.status_code in [200, 201]:
                nodes_data = response.json()
                folders = []
                
                # Extract folder titles and IDs for root folders only (parent_id is None)
                for node in nodes_data:
                    if node.get("node_type") == "folder" and node.get("parent_id") is None:
                        folders.append(node.get("title"))
                
                return {
                    "success": True,
                    "message": f"Found {len(folders)} root folders",
                    "folders": folders
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to get root folders: {str(e)}"
        }

async def get_node_children(node_id: str, node_type: str = "", auth_token: str = "", current_node_id: str = "") -> dict:
    """Get immediate children of a specific node (optionally filtered by node type)"""
    try:
        import httpx
        
        # Get auth token if not provided
        if not auth_token:
            auth_token = await get_auth_token()
        
        print(f"👶 MCP DEBUG - get_node_children called:")
        print(f"   Node ID: {node_id}")
        print(f"   Node type filter: {node_type}")
        print(f"   Auth token present: {bool(auth_token)}")
        
        if not auth_token:
            return {"success": False, "error": "No authentication token provided"}
        
        if not node_id:
            return {"success": False, "error": "Node ID is required"}
        
        # FastGTD API endpoint - get children of specific node
        url = f"{FASTGTD_API_URL}/nodes/"
    
    except Exception as e:
        print(f"❌ MCP ERROR in setup: {str(e)}")
        return {
            "success": False,
            "error": f"MCP tool setup failed: {str(e)}"
        }
    
    # Prepare headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_token}"
    }
    
    # Prepare query parameters
    params = {"parent_id": node_id, "limit": 1000}
    if node_type:
        params["node_type"] = node_type
    
    try:
        async with httpx.AsyncClient() as client:
            # Get children of the specified node
            response = await client.get(
                url,
                headers=headers,
                params=params
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
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to get node children: {str(e)}"
        }

async def get_folder_id(folder_name: str, auth_token: str = "", current_node_id: str = "") -> dict:
    """Get folder ID by folder name - useful for finding the specific folder to work with"""
    logger.info(f"🔍 get_folder_id CALLED - folder_name='{folder_name}', auth_token_present={bool(auth_token)}")
    
    try:
        import httpx
        
        # Get auth token if not provided
        if not auth_token:
            auth_token = await get_auth_token()
        
        print(f"🔍 MCP DEBUG - get_folder_id called:")
        print(f"   Folder name: {folder_name}")
        print(f"   Auth token present: {bool(auth_token)}")
        
        if not auth_token:
            return {"success": False, "error": "No authentication token provided"}
        
        if not folder_name:
            return {"success": False, "error": "Folder name is required"}
        
        # FastGTD API endpoint - get all notes (folders are notes with "Container folder" body)
        url = f"{FASTGTD_API_URL}/nodes/"
    
    except Exception as e:
        print(f"❌ MCP ERROR in setup: {str(e)}")
        return {
            "success": False,
            "error": f"MCP tool setup failed: {str(e)}"
        }
    
    # Prepare headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_token}"
    }
    
    try:
        async with httpx.AsyncClient() as client:
            # Get all folders (folders are their own node type)
            response = await client.get(
                url,
                headers=headers,
                params={"node_type": "folder", "limit": 1000}  # Get all folders
            )
            
            if response.status_code in [200, 201]:
                nodes_data = response.json()
                
                # Find folder with matching name (case-insensitive)
                folder_name_lower = folder_name.lower().strip()
                for node in nodes_data:
                    if node.get("node_type") == "folder":
                        node_title = node.get("title", "").lower().strip()
                        if node_title == folder_name_lower:
                            result = {
                                "success": True,
                                "message": f"Found folder '{folder_name}'",
                                "folder_id": node.get("id"),
                                "folder_name": node.get("title")
                            }
                            logger.info(f"🔍 get_folder_id SUCCESS - found folder_id={result['folder_id']} for '{folder_name}'")
                            return result
                
                # If exact match not found, check for partial matches
                partial_matches = []
                for node in nodes_data:
                    if node.get("node_type") == "folder":
                        node_title = node.get("title", "").lower().strip()
                        if folder_name_lower in node_title or node_title in folder_name_lower:
                            partial_matches.append({
                                "id": node.get("id"),
                                "title": node.get("title")
                            })
                
                if partial_matches:
                    result = {
                        "success": False,
                        "error": f"No exact match found for '{folder_name}', but found similar folders",
                        "suggestions": partial_matches
                    }
                    logger.warning(f"🔍 get_folder_id PARTIAL MATCH - no exact match for '{folder_name}', found {len(partial_matches)} similar")
                    return result
                else:
                    result = {
                        "success": False,
                        "error": f"No folder found with name '{folder_name}'"
                    }
                    logger.error(f"🔍 get_folder_id FAILED - no folder found for '{folder_name}'")
                    return result
            else:
                return {
                    "success": False,
                    "error": f"Failed to retrieve folders: HTTP {response.status_code}",
                    "details": response.text
                }
                
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to find folder: {str(e)}"
        }

async def add_task_to_node_id(node_id: str, task_title: str, description: str = "", priority: str = "medium", auth_token: str = "", current_node_id: str = "") -> dict:
    """Add a task to a specific node by node ID and return the new task's ID"""
    logger.info(f"🎯 add_task_to_node_id CALLED - node_id='{node_id}', task_title='{task_title}', description='{description}', priority='{priority}', auth_token_present={bool(auth_token)}")
    
    try:
        import httpx
        
        # Get auth token if not provided
        if not auth_token:
            auth_token = await get_auth_token()
        
        print(f"🎯 MCP DEBUG - add_task_to_node_id called:")
        print(f"   Node ID: {node_id}")
        print(f"   Task title: {task_title}")
        print(f"   Description: {description}")
        print(f"   Priority: {priority}")
        print(f"   Auth token present: {bool(auth_token)}")
        
        if not auth_token:
            return {"success": False, "error": "No authentication token provided"}
        
        if not node_id:
            return {"success": False, "error": "Node ID is required"}
            
        if not task_title:
            return {"success": False, "error": "Task title is required"}
        
        # FastGTD API endpoint
        url = f"{FASTGTD_API_URL}/nodes/"
    
    except Exception as e:
        print(f"❌ MCP ERROR in setup: {str(e)}")
        return {
            "success": False,
            "error": f"MCP tool setup failed: {str(e)}"
        }
    
    # Create task payload for unified node system
    task_payload = {
        "node_type": "task",
        "title": task_title,
        "parent_id": node_id,  # Use the provided node ID
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
    
    print(f"📤 Final task payload: {task_payload}")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json=task_payload,
                headers=headers
            )
            
            if response.status_code in [200, 201]:
                task_data = response.json()
                task_id = task_data.get("id")
                result = {
                    "success": True,
                    "message": f"Task '{task_title}' added to node successfully",
                    "task_id": task_id,
                    "node_id": node_id,
                    "task_title": task_title
                }
                logger.info(f"🎯 add_task_to_node_id SUCCESS - created task_id={task_id} in node_id={node_id}")
                return result
            else:
                result = {
                    "success": False,
                    "error": f"Failed to add task: HTTP {response.status_code}",
                    "details": response.text
                }
                logger.error(f"🎯 add_task_to_node_id FAILED - HTTP {response.status_code} for task '{task_title}' to node {node_id}")
                return result
                
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to add task to node: {str(e)}"
        }

async def get_node_tree(root_id: str = "", max_depth: int = 10, auth_token: str = "", current_node_id: str = "") -> dict:
    """Get the node tree structure starting from a root node (or from root if no ID provided)"""
    logger.info(f"🌳 get_node_tree CALLED - root_id='{root_id}', max_depth={max_depth}, auth_token_present={bool(auth_token)}")
    
    try:
        import httpx
        
        # Get auth token if not provided
        if not auth_token:
            auth_token = await get_auth_token()
        
        print(f"🌳 MCP DEBUG - get_node_tree called:")
        print(f"   Root ID: {root_id}")
        print(f"   Max depth: {max_depth}")
        print(f"   Auth token present: {bool(auth_token)}")
        
        if not auth_token:
            return {"success": False, "error": "No authentication token provided"}
        
        # FastGTD API endpoint for tree - the API expects root_id as path param or None
        if root_id:
            url = f"{FASTGTD_API_URL}/nodes/tree/{root_id}"
        else:
            # For root/no specific node, we'll need to call the root endpoint
            # First let's try to get nodes at the root level
            url = f"{FASTGTD_API_URL}/nodes/"
    
    except Exception as e:
        print(f"❌ MCP ERROR in setup: {str(e)}")
        return {
            "success": False,
            "error": f"MCP tool setup failed: {str(e)}"
        }
    
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
        # For listing nodes, use different params - don't send parent_id at all for root
        params["limit"] = 100  # Reasonable limit
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                headers=headers,
                params=params
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if root_id:
                    # This is tree format
                    result = {
                        "success": True,
                        "message": f"Retrieved node tree from {root_id} (depth: {max_depth})",
                        "tree": data,
                        "root_id": root_id,
                        "item_count": data.get("total_count", 0),
                        "format": "tree"
                    }
                    logger.info(f"🌳 get_node_tree SUCCESS - retrieved tree with {result['item_count']} items")
                else:
                    # This is a list of root nodes
                    result = {
                        "success": True,
                        "message": f"Retrieved root level nodes",
                        "nodes": data,
                        "root_id": "root",
                        "item_count": len(data) if isinstance(data, list) else 1,
                        "format": "list"
                    }
                    logger.info(f"🌳 get_node_tree SUCCESS - retrieved {result['item_count']} root nodes")
                
                return result
            else:
                result = {
                    "success": False,
                    "error": f"Failed to get node tree: HTTP {response.status_code}",
                    "details": response.text
                }
                logger.error(f"🌳 get_node_tree FAILED - HTTP {response.status_code}")
                return result
                
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to get node tree: {str(e)}"
        }

async def search_nodes(query: str, node_type: str = "", limit: int = 50, auth_token: str = "", current_node_id: str = "") -> dict:
    """Search for nodes by title and content - perfect for finding specific tasks, notes, or folders"""
    logger.info(f"🔍 search_nodes CALLED - query='{query}', node_type='{node_type}', limit={limit}, auth_token_present={bool(auth_token)}")
    
    try:
        import httpx
        
        # Get auth token if not provided
        if not auth_token:
            auth_token = await get_auth_token()
        
        print(f"🔍 MCP DEBUG - search_nodes called:")
        print(f"   Query: {query}")
        print(f"   Node type filter: {node_type}")
        print(f"   Limit: {limit}")
        print(f"   Auth token present: {bool(auth_token)}")
        
        if not auth_token:
            return {"success": False, "error": "No authentication token provided"}
        
        if not query or len(query.strip()) < 1:
            return {"success": False, "error": "Search query is required and must be at least 1 character"}
        
        # FastGTD API endpoint for searching nodes
        url = f"{FASTGTD_API_URL}/nodes/"
    
    except Exception as e:
        print(f"❌ MCP ERROR in setup: {str(e)}")
        return {
            "success": False,
            "error": f"MCP tool setup failed: {str(e)}"
        }
    
    # Prepare headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_token}"
    }
    
    # Query parameters for search
    params = {
        "search": query.strip(),
        "limit": min(limit, 100)  # Cap at 100 for performance
    }
    
    # Add node type filter if specified
    if node_type and node_type in ["task", "note", "folder", "smart_folder", "template"]:
        params["node_type"] = node_type
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                headers=headers,
                params=params
            )
            
            if response.status_code == 200:
                results = response.json()
                result_count = len(results) if isinstance(results, list) else 1
                
                result = {
                    "success": True,
                    "message": f"Found {result_count} result(s) for '{query}'",
                    "query": query,
                    "node_type_filter": node_type or "all types",
                    "results": results,
                    "result_count": result_count
                }
                logger.info(f"🔍 search_nodes SUCCESS - found {result_count} results for '{query}'")
                return result
            else:
                result = {
                    "success": False,
                    "error": f"Failed to search nodes: HTTP {response.status_code}",
                    "details": response.text
                }
                logger.error(f"🔍 search_nodes FAILED - HTTP {response.status_code} for query '{query}'")
                return result
                
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to search nodes: {str(e)}"
        }

async def create_task(title: str, description: str = "", priority: str = "medium", parent_id: str = "", auth_token: str = "", current_node_id: str = "") -> dict:
    """Create a new task with simplified interface - auto-detects best location"""
    logger.info(f"🧪 create_task CALLED - title='{title}', description='{description}', priority='{priority}', parent_id='{parent_id}', auth_token_present={bool(auth_token)}")
    
    try:
        import httpx
        
        # Get auth token if not provided
        if not auth_token:
            auth_token = await get_auth_token()
        
        print(f"🧪 MCP DEBUG - create_task called:")
        print(f"   Title: {title}")
        print(f"   Description: {description}")
        print(f"   Priority: {priority}")
        print(f"   Parent ID: {parent_id}")
        print(f"   Auth token present: {bool(auth_token)}")
        
        # FastGTD API endpoint
        url = f"{FASTGTD_API_URL}/nodes/"
    
    except Exception as e:
        print(f"❌ MCP ERROR in setup: {str(e)}")
        return {
            "success": False,
            "error": f"MCP tool setup failed: {str(e)}"
        }
    
    # Create task payload for unified node system
    task_payload = {
        "node_type": "task",
        "title": title,
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
        print(f"🎯 Using current node as parent: {current_node_id}")
    else:
        # Try to get user's default node (inbox)
        print("🎯 No parent specified, trying to get default node...")
    
    # Prepare headers
    headers = {"Content-Type": "application/json"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    
    try:
        async with httpx.AsyncClient() as client:
            # If no parent specified, get default node
            if not parent_id and not current_node_id and auth_token:
                default_node_response = await client.get(
                    f"{FASTGTD_API_URL}/settings/default-node",
                    headers=headers
                )
                
                if default_node_response.status_code == 200:
                    default_data = default_node_response.json()
                    default_node_id = default_data.get("node_id")
                    if default_node_id:
                        task_payload["parent_id"] = default_node_id
                        print(f"🎯 Using default node as parent: {default_node_id}")
            
            # Create the task
            response = await client.post(url, json=task_payload, headers=headers)
            
            print(f"📡 API Response status: {response.status_code}")
            
            if response.status_code in [200, 201]:
                task_data = response.json()
                print(f"✅ Task created successfully: {task_data}")
                
                return {
                    "success": True,
                    "message": f"Task '{title}' created successfully",
                    "task_id": task_data.get("id"),
                    "task": task_data
                }
            else:
                error_msg = f"Failed to create task: HTTP {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg += f" - {error_data}"
                except:
                    error_msg += f" - {response.text}"
                
                print(f"❌ {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }
                
    except Exception as e:
        error_msg = f"Failed to create task: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            "success": False,
            "error": error_msg
        }

async def update_task(task_id: str, title: str = "", description: str = "", priority: str = "", status: str = "", due_at: str = "", earliest_start_at: str = "", archived: bool = None, recurrence_rule: str = "", recurrence_anchor: str = "", auth_token: str = "", current_node_id: str = "") -> dict:
    """Update an existing task's properties"""
    logger.info(f"🧪 update_task CALLED - task_id='{task_id}', title='{title}', description='{description}', priority='{priority}', status='{status}', auth_token_present={bool(auth_token)}")
    
    try:
        import httpx
        
        # Get auth token if not provided
        if not auth_token:
            auth_token = await get_auth_token()
        
        print(f"🧪 MCP DEBUG - update_task called:")
        print(f"   Task ID: {task_id}")
        print(f"   Title: {title}")
        print(f"   Description: {description}")
        print(f"   Priority: {priority}")
        print(f"   Status: {status}")
        print(f"   Auth token present: {bool(auth_token)}")
        
        # FastGTD API endpoint
        url = f"{FASTGTD_API_URL}/nodes/{task_id}"
    
    except Exception as e:
        print(f"❌ MCP ERROR in setup: {str(e)}")
        return {
            "success": False,
            "error": f"MCP tool setup failed: {str(e)}"
        }
    
    # Build update payload - only include fields that are provided
    update_payload = {}
    
    if title:
        update_payload["title"] = title
    
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
    if archived is not None:  # Allow setting to False
        task_data_updates["archived"] = archived
    if recurrence_rule:
        task_data_updates["recurrence_rule"] = recurrence_rule
    if recurrence_anchor:
        task_data_updates["recurrence_anchor"] = recurrence_anchor
    
    if task_data_updates:
        update_payload["task_data"] = task_data_updates
    
    # Check if we have anything to update
    if not update_payload:
        return {
            "success": False,
            "error": "No fields provided to update. Please specify title, description, priority, or status."
        }
    
    # Prepare headers
    headers = {"Content-Type": "application/json"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    
    try:
        async with httpx.AsyncClient() as client:
            # Update the task
            response = await client.put(url, json=update_payload, headers=headers)
            
            print(f"📡 API Response status: {response.status_code}")
            
            if response.status_code == 200:
                task_data = response.json()
                print(f"✅ Task updated successfully: {task_data}")
                
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
                error_msg = f"Failed to update task: HTTP {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg += f" - {error_data}"
                except:
                    error_msg += f" - {response.text}"
                
                print(f"❌ {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }
                
    except Exception as e:
        error_msg = f"Failed to update task: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            "success": False,
            "error": error_msg
        }

async def complete_task(task_id: str, auth_token: str = "", current_node_id: str = "") -> dict:
    """Mark a task as completed"""
    logger.info(f"✅ complete_task CALLED - task_id='{task_id}', auth_token_present={bool(auth_token)}")
    
    try:
        import httpx
        
        # Get auth token if not provided
        if not auth_token:
            auth_token = await get_auth_token()
        
        print(f"✅ MCP DEBUG - complete_task called:")
        print(f"   Task ID: {task_id}")
        print(f"   Auth token present: {bool(auth_token)}")
        
        if not task_id:
            return {"success": False, "error": "Task ID is required"}
        
        # FastGTD API endpoint
        url = f"{FASTGTD_API_URL}/nodes/{task_id}"
    
    except Exception as e:
        print(f"❌ MCP ERROR in setup: {str(e)}")
        return {
            "success": False,
            "error": f"MCP tool setup failed: {str(e)}"
        }
    
    # Update payload to mark as done
    update_payload = {
        "task_data": {
            "status": "done"
        }
    }
    
    # Prepare headers
    headers = {"Content-Type": "application/json"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    
    try:
        async with httpx.AsyncClient() as client:
            # Update the task status to done
            response = await client.put(url, json=update_payload, headers=headers)
            
            print(f"📡 API Response status: {response.status_code}")
            
            if response.status_code == 200:
                task_data = response.json()
                print(f"✅ Task completed successfully: {task_data}")
                
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
                error_msg = f"Failed to complete task: HTTP {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg += f" - {error_data}"
                except:
                    error_msg += f" - {response.text}"
                
                print(f"❌ {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }
                
    except Exception as e:
        error_msg = f"Failed to complete task: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            "success": False,
            "error": error_msg
        }

async def delete_task(task_id: str, auth_token: str = "", current_node_id: str = "") -> dict:
    """Delete a task permanently"""
    logger.info(f"🗑️ delete_task CALLED - task_id='{task_id}', auth_token_present={bool(auth_token)}")
    
    try:
        import httpx
        
        # Get auth token if not provided
        if not auth_token:
            auth_token = await get_auth_token()
        
        print(f"🗑️ MCP DEBUG - delete_task called:")
        print(f"   Task ID: {task_id}")
        print(f"   Auth token present: {bool(auth_token)}")
        
        if not task_id:
            return {"success": False, "error": "Task ID is required"}
        
        # FastGTD API endpoint
        url = f"{FASTGTD_API_URL}/nodes/{task_id}"
    
    except Exception as e:
        print(f"❌ MCP ERROR in setup: {str(e)}")
        return {
            "success": False,
            "error": f"MCP tool setup failed: {str(e)}"
        }
    
    # Prepare headers
    headers = {"Content-Type": "application/json"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    
    try:
        async with httpx.AsyncClient() as client:
            # Delete the task
            response = await client.delete(url, headers=headers)
            
            print(f"📡 API Response status: {response.status_code}")
            
            if response.status_code == 200:
                print(f"🗑️ Task deleted successfully")
                
                return {
                    "success": True,
                    "message": f"Task '{task_id}' deleted successfully"
                }
            else:
                error_msg = f"Failed to delete task: HTTP {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg += f" - {error_data}"
                except:
                    error_msg += f" - {response.text}"
                
                print(f"❌ {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }
                
    except Exception as e:
        error_msg = f"Failed to delete task: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            "success": False,
            "error": error_msg
        }

async def create_folder(title: str, parent_id: str = "", auth_token: str = "", current_node_id: str = "") -> dict:
    """Create a new folder - auto-detects best location (current folder or root)"""
    logger.info(f"📁 create_folder CALLED - title='{title}', parent_id='{parent_id}', auth_token_present={bool(auth_token)}")
    
    try:
        import httpx
        
        # Get auth token if not provided
        if not auth_token:
            auth_token = await get_auth_token()
        
        print(f"📁 MCP DEBUG - create_folder called:")
        print(f"   Title: {title}")
        print(f"   Parent ID: {parent_id}")
        print(f"   Current node ID: {current_node_id}")
        print(f"   Auth token present: {bool(auth_token)}")
        
        if not title:
            return {"success": False, "error": "Folder title is required"}
        
        # FastGTD API endpoint
        url = f"{FASTGTD_API_URL}/nodes/"
    
    except Exception as e:
        print(f"❌ MCP ERROR in setup: {str(e)}")
        return {
            "success": False,
            "error": f"MCP tool setup failed: {str(e)}"
        }
    
    # Create folder payload - folders are their own node type
    folder_payload = {
        "node_type": "folder",
        "title": title
    }
    
    # Determine parent location
    if parent_id:
        folder_payload["parent_id"] = parent_id
        print(f"🎯 Using specified parent: {parent_id}")
    elif current_node_id:
        folder_payload["parent_id"] = current_node_id
        print(f"🎯 Using current node as parent: {current_node_id}")
    else:
        # Try to get user's default node or create at root
        print("🎯 No parent specified, trying to get default node...")
    
    # Prepare headers
    headers = {"Content-Type": "application/json"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    
    try:
        async with httpx.AsyncClient() as client:
            # If no parent specified, get default node
            if not parent_id and not current_node_id and auth_token:
                default_node_response = await client.get(
                    f"{FASTGTD_API_URL}/settings/default-node",
                    headers=headers
                )
                
                if default_node_response.status_code == 200:
                    default_data = default_node_response.json()
                    default_node_id = default_data.get("node_id")
                    if default_node_id:
                        folder_payload["parent_id"] = default_node_id
                        print(f"🎯 Using default node as parent: {default_node_id}")
            
            print(f"📤 Final folder payload: {folder_payload}")
            
            # Create the folder
            response = await client.post(url, json=folder_payload, headers=headers)
            
            print(f"📡 API Response status: {response.status_code}")
            
            if response.status_code in [200, 201]:
                folder_data = response.json()
                print(f"✅ Folder created successfully: {folder_data}")
                
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
                error_msg = f"Failed to create folder: HTTP {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg += f" - {error_data}"
                except:
                    error_msg += f" - {response.text}"
                
                print(f"❌ {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }
                
    except Exception as e:
        error_msg = f"Failed to create folder: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            "success": False,
            "error": error_msg
        }

async def move_node(node_id: str, new_parent_id: str = "", new_sort_order: int = None, auth_token: str = "", current_node_id: str = "") -> dict:
    """Move a task or note to a different folder (or to root if no parent specified)"""
    logger.info(f"🔄 move_node CALLED - node_id='{node_id}', new_parent_id='{new_parent_id}', new_sort_order={new_sort_order}, auth_token_present={bool(auth_token)}")
    
    try:
        import httpx
        
        # Get auth token if not provided
        if not auth_token:
            auth_token = await get_auth_token()
        
        print(f"🔄 MCP DEBUG - move_node called:")
        print(f"   Node ID: {node_id}")
        print(f"   New Parent ID: {new_parent_id}")
        print(f"   New Sort Order: {new_sort_order}")
        print(f"   Auth token present: {bool(auth_token)}")
        
        if not node_id:
            return {"success": False, "error": "Node ID is required"}
        
        if not auth_token:
            return {"success": False, "error": "Authentication token is required"}
        
        # FastGTD API endpoint for moving nodes
        url = f"{FASTGTD_API_URL}/nodes/move"
    
    except Exception as e:
        print(f"❌ MCP ERROR in setup: {str(e)}")
        return {
            "success": False,
            "error": f"MCP tool setup failed: {str(e)}"
        }
    
    # Create move payload
    move_payload = {
        "node_id": node_id,
        "new_parent_id": new_parent_id if new_parent_id else None
    }
    
    if new_sort_order is not None:
        move_payload["new_sort_order"] = new_sort_order
    
    # Prepare headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_token}"
    }
    
    print(f"📤 Move payload: {move_payload}")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=move_payload, headers=headers)
            
            print(f"📡 API Response status: {response.status_code}")
            
            if response.status_code in [200, 201]:
                result_data = response.json() if response.content else {"message": "Node moved successfully"}
                print(f"✅ Node moved successfully: {result_data}")
                
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
                error_msg = f"Failed to move node: HTTP {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg += f" - {error_data}"
                except:
                    error_msg += f" - {response.text}"
                
                print(f"❌ {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }
                
    except Exception as e:
        error_msg = f"Failed to move node: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            "success": False,
            "error": error_msg
        }

async def add_tag(node_id: str, tag_name: str, tag_description: str = "", tag_color: str = "", auth_token: str = "", current_node_id: str = "") -> dict:
    """Add a tag to a task, note, or folder (creates tag if it doesn't exist)"""
    logger.info(f"🏷️ add_tag CALLED - node_id='{node_id}', tag_name='{tag_name}', tag_description='{tag_description}', tag_color='{tag_color}', auth_token_present={bool(auth_token)}")
    
    try:
        import httpx
        
        # Get auth token if not provided
        if not auth_token:
            auth_token = await get_auth_token()
        
        print(f"🏷️ MCP DEBUG - add_tag called:")
        print(f"   Node ID: {node_id}")
        print(f"   Tag Name: {tag_name}")
        print(f"   Tag Description: {tag_description}")
        print(f"   Tag Color: {tag_color}")
        print(f"   Auth token present: {bool(auth_token)}")
        
        if not node_id:
            return {"success": False, "error": "Node ID is required"}
        
        if not tag_name:
            return {"success": False, "error": "Tag name is required"}
        
        if not auth_token:
            return {"success": False, "error": "Authentication token is required"}
        
        # FastGTD API endpoints
        create_tag_url = f"{FASTGTD_API_URL}/tags"
        attach_tag_url = f"{FASTGTD_API_URL}/nodes/{node_id}/tags"
    
    except Exception as e:
        print(f"❌ MCP ERROR in setup: {str(e)}")
        return {
            "success": False,
            "error": f"MCP tool setup failed: {str(e)}"
        }
    
    # Prepare headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_token}"
    }
    
    try:
        async with httpx.AsyncClient() as client:
            # Step 1: Create or get tag
            create_params = {"name": tag_name}
            if tag_description:
                create_params["description"] = tag_description
            if tag_color:
                create_params["color"] = tag_color
            
            print(f"📤 Creating/finding tag with params: {create_params}")
            
            tag_response = await client.post(
                create_tag_url,
                headers=headers,
                params=create_params
            )
            
            print(f"📡 Tag creation response status: {tag_response.status_code}")
            
            if tag_response.status_code not in [200, 201]:
                error_msg = f"Failed to create tag: HTTP {tag_response.status_code}"
                try:
                    error_data = tag_response.json()
                    error_msg += f" - {error_data}"
                except:
                    error_msg += f" - {tag_response.text}"
                
                print(f"❌ {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }
            
            tag_data = tag_response.json()
            tag_id = tag_data.get("id")
            print(f"✅ Tag ready: {tag_data}")
            
            if not tag_id:
                return {
                    "success": False,
                    "error": "Failed to get tag ID from response"
                }
            
            # Step 2: Attach tag to node
            attach_url = f"{attach_tag_url}/{tag_id}"
            print(f"📤 Attaching tag to node: {attach_url}")
            
            attach_response = await client.post(attach_url, headers=headers)
            
            print(f"📡 Tag attachment response status: {attach_response.status_code}")
            
            if attach_response.status_code in [200, 201]:
                print(f"✅ Tag '{tag_name}' attached to node successfully")
                
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
                error_msg = f"Failed to attach tag to node: HTTP {attach_response.status_code}"
                try:
                    error_data = attach_response.json()
                    error_msg += f" - {error_data}"
                except:
                    error_msg += f" - {attach_response.text}"
                
                print(f"❌ {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }
                
    except Exception as e:
        error_msg = f"Failed to add tag: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            "success": False,
            "error": error_msg
        }

async def remove_tag(node_id: str, tag_name: str, auth_token: str = "", current_node_id: str = "") -> dict:
    """Remove a tag from a task, note, or folder"""
    logger.info(f"🏷️❌ remove_tag CALLED - node_id='{node_id}', tag_name='{tag_name}', auth_token_present={bool(auth_token)}")
    
    try:
        import httpx
        
        # Get auth token if not provided
        if not auth_token:
            auth_token = await get_auth_token()
        
        print(f"🏷️❌ MCP DEBUG - remove_tag called:")
        print(f"   Node ID: {node_id}")
        print(f"   Tag Name: {tag_name}")
        print(f"   Auth token present: {bool(auth_token)}")
        
        if not node_id:
            return {"success": False, "error": "Node ID is required"}
        
        if not tag_name:
            return {"success": False, "error": "Tag name is required"}
        
        if not auth_token:
            return {"success": False, "error": "Authentication token is required"}
        
        # FastGTD API endpoints
        get_tags_url = f"{FASTGTD_API_URL}/nodes/{node_id}/tags"
        detach_tag_url = f"{FASTGTD_API_URL}/nodes/{node_id}/tags"
    
    except Exception as e:
        print(f"❌ MCP ERROR in setup: {str(e)}")
        return {
            "success": False,
            "error": f"MCP tool setup failed: {str(e)}"
        }
    
    # Prepare headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_token}"
    }
    
    try:
        async with httpx.AsyncClient() as client:
            # Step 1: Get all tags for the node to find the tag ID
            print(f"📤 Getting tags for node: {get_tags_url}")
            
            tags_response = await client.get(get_tags_url, headers=headers)
            
            print(f"📡 Get tags response status: {tags_response.status_code}")
            
            if tags_response.status_code not in [200]:
                error_msg = f"Failed to get node tags: HTTP {tags_response.status_code}"
                try:
                    error_data = tags_response.json()
                    error_msg += f" - {error_data}"
                except:
                    error_msg += f" - {tags_response.text}"
                
                print(f"❌ {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }
            
            tags_data = tags_response.json()
            print(f"✅ Node tags: {tags_data}")
            
            # Find the tag with matching name (case-insensitive)
            tag_id = None
            tag_name_lower = tag_name.lower().strip()
            
            for tag in tags_data:
                if tag.get("name", "").lower().strip() == tag_name_lower:
                    tag_id = tag.get("id")
                    break
            
            if not tag_id:
                return {
                    "success": False,
                    "error": f"Tag '{tag_name}' not found on this node",
                    "available_tags": [tag.get("name") for tag in tags_data if tag.get("name")]
                }
            
            # Step 2: Remove tag from node
            detach_url = f"{detach_tag_url}/{tag_id}"
            print(f"📤 Removing tag from node: {detach_url}")
            
            detach_response = await client.delete(detach_url, headers=headers)
            
            print(f"📡 Tag removal response status: {detach_response.status_code}")
            
            if detach_response.status_code in [204]:
                print(f"✅ Tag '{tag_name}' removed from node successfully")
                
                return {
                    "success": True,
                    "message": f"Tag '{tag_name}' removed from node successfully",
                    "node_id": node_id,
                    "tag_name": tag_name,
                    "tag_id": tag_id
                }
            else:
                error_msg = f"Failed to remove tag from node: HTTP {detach_response.status_code}"
                try:
                    error_data = detach_response.json()
                    error_msg += f" - {error_data}"
                except:
                    error_msg += f" - {detach_response.text}"
                
                print(f"❌ {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }
                
    except Exception as e:
        error_msg = f"Failed to remove tag: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            "success": False,
            "error": error_msg
        }


async def get_today_tasks(auth_token: str = "", current_node_id: str = "") -> dict:
    """Get all tasks that are due today."""
    try:
        print("🔍 MCP TOOL: get_today_tasks")
        print(f"   Auth token present: {bool(auth_token)}")
        
        if not auth_token:
            auth_token = await get_auth_token()
        
        # Get today's date in ISO format (start and end of day)
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).date()
        today_start = datetime.combine(today, datetime.min.time()).replace(tzinfo=timezone.utc).isoformat()
        today_end = datetime.combine(today, datetime.max.time()).replace(tzinfo=timezone.utc).isoformat()
        
        print(f"   Looking for tasks due between: {today_start} and {today_end}")
        
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
        
        import httpx
        
        # Get auth token if not provided
        if not auth_token:
            auth_token = await get_auth_token()
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                headers=headers,
                params=params
            )
            
            if response.status_code == 200:
                all_tasks = response.json()
                today_tasks = []
                
                for task in all_tasks:
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
            else:
                error_msg = f"API request failed: HTTP {response.status_code}"
                if response.status_code == 401:
                    error_msg = "Authentication failed - invalid token"
                elif response.status_code == 404:
                    error_msg = "Tasks endpoint not found"
                    
                return {"success": False, "error": error_msg}
                
    except Exception as e:
        print(f"❌ MCP ERROR in get_today_tasks: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to get today's tasks: {str(e)}"
        }


async def get_overdue_tasks(auth_token: str = "", current_node_id: str = "") -> dict:
    """Get all tasks that are overdue (due date in the past)."""
    try:
        print("🔍 MCP TOOL: get_overdue_tasks")
        print(f"   Auth token present: {bool(auth_token)}")
        
        if not auth_token:
            auth_token = await get_auth_token()
        
        # Get current datetime
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        
        print(f"   Looking for tasks due before: {now.isoformat()}")
        
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
        
        import httpx
        
        # Get auth token if not provided
        if not auth_token:
            auth_token = await get_auth_token()
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                headers=headers,
                params=params
            )
            
            if response.status_code == 200:
                all_tasks = response.json()
                overdue_tasks = []
                
                for task in all_tasks:
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
            else:
                error_msg = f"API request failed: HTTP {response.status_code}"
                if response.status_code == 401:
                    error_msg = "Authentication failed - invalid token"
                elif response.status_code == 404:
                    error_msg = "Tasks endpoint not found"
                    
                return {"success": False, "error": error_msg}
                
    except Exception as e:
        print(f"❌ MCP ERROR in get_overdue_tasks: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to get overdue tasks: {str(e)}"
        }


async def update_note(note_id: str, title: str = "", content: str = "", auth_token: str = "", current_node_id: str = "") -> dict:
    """Update an existing note's title and/or content."""
    try:
        print("✏️ MCP TOOL: update_note")
        print(f"   Note ID: {note_id}")
        print(f"   Title: {title}")
        print(f"   Content length: {len(content) if content else 0} characters")
        print(f"   Auth token present: {bool(auth_token)}")
        
        if not auth_token:
            auth_token = await get_auth_token()
        
        if not note_id:
            return {"success": False, "error": "Note ID is required"}
            
        # Must provide either title or content to update
        if not title and not content:
            return {"success": False, "error": "Must provide either title or content to update"}
        
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
        
        print(f"   Update payload: {update_data}")
        
        import httpx
        
        # Get auth token if not provided
        if not auth_token:
            auth_token = await get_auth_token()
        async with httpx.AsyncClient() as client:
            response = await client.put(
                url,
                headers=headers,
                json=update_data
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
                error_msg = f"API request failed: HTTP {response.status_code}"
                if response.status_code == 401:
                    error_msg = "Authentication failed - invalid token"
                elif response.status_code == 404:
                    error_msg = "Note not found - it may have been deleted"
                elif response.status_code == 400:
                    try:
                        error_detail = response.json()
                        error_msg = f"Bad request: {error_detail.get('detail', 'Unknown error')}"
                    except:
                        error_msg = "Bad request - invalid note data"
                    
                return {"success": False, "error": error_msg}
                
    except Exception as e:
        print(f"❌ MCP ERROR in update_note: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to update note: {str(e)}"
        }


async def get_smart_folder_contents(smart_folder_id: str, limit: int = 100, offset: int = 0, auth_token: str = "", current_node_id: str = "") -> dict:
    """Get the contents of a smart folder by evaluating its rules."""
    try:
        print("🤖 MCP TOOL: get_smart_folder_contents")
        print(f"   Smart folder ID: {smart_folder_id}")
        print(f"   Limit: {limit}")
        print(f"   Offset: {offset}")
        print(f"   Auth token present: {bool(auth_token)}")
        
        if not auth_token:
            auth_token = await get_auth_token()
        
        if not smart_folder_id:
            return {"success": False, "error": "Smart folder ID is required"}
        
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
        
        import httpx
        
        # Get auth token if not provided
        if not auth_token:
            auth_token = await get_auth_token()
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                headers=headers,
                params=params
            )
            
            if response.status_code == 200:
                contents = response.json()
                
                # Extract relevant information from each matching node
                processed_contents = []
                for node in contents:
                    node_info = {
                        'id': node['id'],
                        'title': node['title'],
                        'node_type': node['node_type'],
                        'created_at': node['created_at'],
                        'updated_at': node['updated_at'],
                        'parent_id': node.get('parent_id'),
                        'tags': [tag['name'] for tag in node.get('tags', [])]
                    }
                    
                    # Add type-specific data
                    if node['node_type'] == 'task':
                        task_data = node.get('task_data', {})
                        node_info.update({
                            'status': task_data.get('status'),
                            'priority': task_data.get('priority'),
                            'due_at': task_data.get('due_at'),
                            'description': task_data.get('description', '')[:100] + ('...' if len(task_data.get('description', '')) > 100 else '')
                        })
                    elif node['node_type'] == 'note':
                        note_data = node.get('note_data', {})
                        node_info.update({
                            'content_preview': note_data.get('body', '')[:100] + ('...' if len(note_data.get('body', '')) > 100 else '')
                        })
                
                    processed_contents.append(node_info)
                
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
                error_msg = f"API request failed: HTTP {response.status_code}"
                if response.status_code == 401:
                    error_msg = "Authentication failed - invalid token"
                elif response.status_code == 404:
                    error_msg = "Smart folder not found"
                elif response.status_code == 400:
                    try:
                        error_detail = response.json()
                        error_msg = f"Bad request: {error_detail.get('detail', 'Invalid smart folder request')}"
                    except:
                        error_msg = "Bad request - invalid smart folder ID"
                    
                return {"success": False, "error": error_msg}
                
    except Exception as e:
        print(f"❌ MCP ERROR in get_smart_folder_contents: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to get smart folder contents: {str(e)}"
        }


async def instantiate_template(template_id: str, name: str, parent_id: str = "", auth_token: str = "", current_node_id: str = "") -> dict:
    """Create a new instance from a template with all its contents."""
    try:
        print("📋 MCP TOOL: instantiate_template")
        print(f"   Template ID: {template_id}")
        print(f"   Instance name: {name}")
        print(f"   Parent ID: {parent_id}")
        print(f"   Auth token present: {bool(auth_token)}")
        
        if not auth_token:
            return {"success": False, "error": "No authentication token provided"}
        
        if not template_id:
            return {"success": False, "error": "Template ID is required"}
            
        if not name:
            return {"success": False, "error": "Instance name is required"}
        
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
        
        print(f"   Query parameters: {params}")
        
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                headers=headers,
                params=params
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
                error_msg = f"API request failed: HTTP {response.status_code}"
                if response.status_code == 401:
                    error_msg = "Authentication failed - invalid token"
                elif response.status_code == 404:
                    error_msg = "Template not found - it may have been deleted"
                elif response.status_code == 400:
                    try:
                        error_detail = response.json()
                        error_msg = f"Bad request: {error_detail.get('detail', 'Unknown error')}"
                    except:
                        error_msg = "Bad request - invalid template or parameters"
                    
                return {"success": False, "error": error_msg}
                
    except Exception as e:
        print(f"❌ MCP ERROR in instantiate_template: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to instantiate template: {str(e)}"
        }


async def list_templates(category: str = "", limit: int = 50, offset: int = 0, auth_token: str = "", current_node_id: str = "") -> dict:
    """List all available templates with optional category filter."""
    try:
        print("📋 MCP TOOL: list_templates")
        print(f"   Category filter: {category}")
        print(f"   Limit: {limit}")
        print(f"   Offset: {offset}")
        print(f"   Auth token present: {bool(auth_token)}")
        
        if not auth_token:
            return {"success": False, "error": "No authentication token provided"}
        
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
        
        print(f"   Query parameters: {params}")
        
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                headers=headers,
                params=params
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
                error_msg = f"API request failed: HTTP {response.status_code}"
                if response.status_code == 401:
                    error_msg = "Authentication failed - invalid token"
                elif response.status_code == 400:
                    try:
                        error_detail = response.json()
                        error_msg = f"Bad request: {error_detail.get('detail', 'Invalid request')}"
                    except:
                        error_msg = "Bad request - invalid parameters"
                    
                return {"success": False, "error": error_msg}
                
    except Exception as e:
        print(f"❌ MCP ERROR in list_templates: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to list templates: {str(e)}"
        }


async def search_templates(query: str, category: str = "", limit: int = 50, auth_token: str = "", current_node_id: str = "") -> dict:
    """Search for templates by name or description."""
    try:
        print("🔍 MCP TOOL: search_templates")
        print(f"   Search query: {query}")
        print(f"   Category filter: {category}")
        print(f"   Limit: {limit}")
        print(f"   Auth token present: {bool(auth_token)}")
        
        if not auth_token:
            return {"success": False, "error": "No authentication token provided"}
        
        if not query or len(query.strip()) < 1:
            return {"success": False, "error": "Search query is required and must be at least 1 character"}
        
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
            "limit": min(limit, 100)  # Cap at 100 for performance
        }
        
        print(f"   Query parameters: {params}")
        
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                headers=headers,
                params=params
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
                error_msg = f"API request failed: HTTP {response.status_code}"
                if response.status_code == 401:
                    error_msg = "Authentication failed - invalid token"
                elif response.status_code == 400:
                    try:
                        error_detail = response.json()
                        error_msg = f"Bad request: {error_detail.get('detail', 'Invalid search query')}"
                    except:
                        error_msg = "Bad request - invalid search parameters"
                    
                return {"success": False, "error": error_msg}
                
    except Exception as e:
        print(f"❌ MCP ERROR in search_templates: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to search templates: {str(e)}"
        }

# Tool handlers mapping
TOOL_HANDLERS = {
    "add_task_to_inbox": add_task_to_inbox,
    "add_task_to_current_node": add_task_to_current_node,
    "add_folder_to_current_node": add_folder_to_current_node,
    "add_note_to_current_node": add_note_to_current_node,
    "get_all_folders": get_all_folders,
    "get_root_folders": get_root_folders,
    "get_node_children": get_node_children,
    "get_folder_id": get_folder_id,
    "add_task_to_node_id": add_task_to_node_id,
    "get_node_tree": get_node_tree,
    "search_nodes": search_nodes,
    "create_task": create_task,
    "update_task": update_task,
    "complete_task": complete_task,
    "delete_task": delete_task,
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
            }
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
            }
        ),
        Tool(
            name="add_folder_to_current_node",
            description="Add a new folder to the user's currently selected node/folder",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Folder name (required)"}
                },
                "required": ["title"]
            }
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
            }
        ),
        Tool(
            name="get_all_folders",
            description="Get all folder names in the user's node tree - useful for AI to help find the right folder when user mentions one",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="get_root_folders",
            description="Get only root-level folders (folders with no parent) - useful for showing top-level organization",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="get_node_children",
            description="Get immediate children of a specific node (optionally filtered by type) - perfect for browsing folder contents or exploring hierarchy",
            inputSchema={
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "ID of the parent node to get children from (required)"},
                    "node_type": {"type": "string", "description": "Filter by node type: task, note, folder, smart_folder, template (optional)"}
                },
                "required": ["node_id"]
            }
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
            }
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
            }
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
            }
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
            }
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
            }
        ),
        Tool(
            name="search_nodes",
            description="Search for nodes by title and content - perfect for finding specific tasks, notes, folders, or any content",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search term to look for in titles and content (required)"},
                    "node_type": {"type": "string", "description": "Filter by node type: 'task', 'note', 'folder', 'smart_folder', or 'template' (optional)"},
                    "limit": {"type": "integer", "description": "Maximum number of results to return (default: 50, max: 100)"}
                },
                "required": ["query"]
            }
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
            }
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
            }
        ),
        Tool(
            name="create_folder",
            description="Create a new folder - automatically finds best location (current folder or root)",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Folder title/name (required)"},
                    "parent_id": {"type": "string", "description": "Specific parent folder ID (optional - auto-detects if not provided)"}
                },
                "required": ["title"]
            }
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
            }
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
            }
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
            }
        ),
        Tool(
            name="get_today_tasks",
            description="Get all tasks that are due today",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="get_overdue_tasks", 
            description="Get all tasks that are overdue (due date in the past)",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
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
            }
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
            }
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
            }
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
            }
        ),
        Tool(
            name="search_templates",
            description="Search for templates by name or description",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query to match template names or descriptions (required)"},
                    "category": {"type": "string", "description": "Filter by category (optional)"},
                    "limit": {"type": "integer", "description": "Maximum number of templates to return (default: 50, max: 100)"}
                },
                "required": ["query"]
            }
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