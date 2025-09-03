#!/usr/bin/env python3
"""
Cleanup script to remove test data created by MCP function tests
Run this periodically to keep your FastGTD instance clean
"""

import asyncio
import sys
import re
from dotenv import load_dotenv

load_dotenv()

async def cleanup_test_folders():
    """Find and delete test folders created during testing"""
    print("🧹 Cleaning up test folders...")
    
    sys.path.append('/home/bkarjoo/dev/gtdmcp')
    try:
        from fastgtd_mcp_server import get_all_folders, get_folder_id, delete_task  # delete_task works for all nodes
        
        # Get all folders
        folders_result = await get_all_folders()
        if not folders_result.get("success"):
            print(f"❌ Failed to get folders: {folders_result}")
            return False
        
        folders = folders_result.get("folders", [])
        
        # Test folder patterns to identify
        test_patterns = [
            r"test.*folder",
            r".*test.*",
            r"debug.*folder", 
            r"tree.*parent",
            r"source.*folder",
            r"destination.*folder",
            r"parent.*folder.*test",
            r"searchable.*folder",
            r"comprehensive.*test.*folder",
            r".*for.*test.*",
            r".*move.*test.*"
        ]
        
        deleted_count = 0
        
        # Check each folder against test patterns
        for folder_name in folders:
            is_test_folder = any(
                re.search(pattern, folder_name.lower()) 
                for pattern in test_patterns
            )
            
            if is_test_folder:
                print(f"🗑️  Found test folder: '{folder_name}'")
                
                # Get the folder ID so we can delete it
                folder_id_result = await get_folder_id(folder_name)
                if folder_id_result.get("success"):
                    folder_id = folder_id_result.get("folder_id")
                    if folder_id:
                        # Delete the folder
                        delete_result = await delete_task(folder_id)
                        if delete_result.get("success"):
                            print(f"   ✅ Deleted folder: {folder_name}")
                            deleted_count += 1
                        else:
                            print(f"   ❌ Failed to delete folder: {delete_result.get('error')}")
                    else:
                        print(f"   ❌ No folder ID returned for: {folder_name}")
                else:
                    print(f"   ❌ Could not get ID for folder: {folder_name}")
        
        print(f"✅ Deleted {deleted_count} test folders")
        return True
        
    except Exception as e:
        print(f"❌ Cleanup failed: {e}")
        return False

async def cleanup_test_tasks():
    """Find and delete test tasks"""
    print("🧹 Cleaning up test tasks...")
    
    sys.path.append('/home/bkarjoo/dev/gtdmcp')
    try:
        from fastgtd_mcp_server import search_nodes, delete_task
        
        # Search for tasks with test-related titles
        test_queries = [
            "test task",
            "task for testing", 
            "debug task",
            "searchable",
            "overdue task",
            "task due today",
            "child task",
            "task to move",
            "task to update",
            "task to complete",
            "task to delete",
            "task for tagging"
        ]
        
        deleted_count = 0
        
        for query in test_queries:
            search_result = await search_nodes(query=query, node_type="task", limit=50)
            
            if search_result.get("success"):
                tasks = search_result.get("results", [])
                
                for task in tasks:
                    task_id = task.get("id")
                    task_title = task.get("title", "")
                    
                    if task_id and task_title:
                        print(f"🗑️  Deleting test task: '{task_title}'")
                        delete_result = await delete_task(task_id)
                        if delete_result.get("success"):
                            deleted_count += 1
                        else:
                            print(f"   ❌ Failed to delete: {delete_result.get('error')}")
        
        print(f"✅ Deleted {deleted_count} test tasks")
        return True
        
    except Exception as e:
        print(f"❌ Task cleanup failed: {e}")
        return False

async def cleanup_test_notes():
    """Find and delete test notes"""
    print("🧹 Cleaning up test notes...")
    
    sys.path.append('/home/bkarjoo/dev/gtdmcp')
    try:
        from fastgtd_mcp_server import search_nodes, delete_task  # delete_task works for notes too
        
        # Search for notes with test-related titles
        search_result = await search_nodes(query="test note", node_type="note", limit=50)
        
        deleted_count = 0
        
        if search_result.get("success"):
            notes = search_result.get("results", [])
            
            for note in notes:
                note_id = note.get("id")
                note_title = note.get("title", "")
                
                if note_id and "test" in note_title.lower():
                    print(f"🗑️  Deleting test note: '{note_title}'")
                    delete_result = await delete_task(note_id)  # Works for any node type
                    if delete_result.get("success"):
                        deleted_count += 1
                    else:
                        print(f"   ❌ Failed to delete: {delete_result.get('error')}")
        
        print(f"✅ Deleted {deleted_count} test notes")
        return True
        
    except Exception as e:
        print(f"❌ Note cleanup failed: {e}")
        return False

async def main():
    import sys
    
    print("FastGTD MCP Test Data Cleanup")
    print("============================")
    print("This script will remove test data created during MCP function testing.")
    
    # Check if running non-interactively (from another script)
    auto_confirm = len(sys.argv) > 1 and sys.argv[1] == "--auto"
    
    if not auto_confirm:
        print("⚠️  Only run this if you want to clean up test artifacts!")
        print()
        
        # Ask for confirmation only in interactive mode
        try:
            confirm = input("Continue with cleanup? (y/N): ").strip().lower()
            if confirm != 'y':
                print("❌ Cleanup cancelled")
                return False
        except (KeyboardInterrupt, EOFError):
            print("\n❌ Cleanup cancelled")
            return False
    else:
        print("🤖 Running in automated mode (--auto flag detected)")
        print()
    
    print("🚀 Starting cleanup...")
    
    # Run all cleanup functions
    task_result = await cleanup_test_tasks()
    note_result = await cleanup_test_notes()  
    folder_result = await cleanup_test_folders()
    
    print("\n📊 Cleanup Summary:")
    print(f"Tasks: {'✅ Cleaned' if task_result else '❌ Failed'}")
    print(f"Notes: {'✅ Cleaned' if note_result else '❌ Failed'}")  
    print(f"Folders: {'ℹ️  Scanned' if folder_result else '❌ Failed'}")
    
    print("\n✅ Cleanup completed!")
    print("💡 Tip: Test folders may need manual cleanup from the FastGTD UI")
    
    return task_result and note_result and folder_result

if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)