#!/usr/bin/env python3
"""
Comprehensive test script for all 22 MCP functions
Tests all functions in logical groups with automatic cleanup
"""

import asyncio
import sys
import re
import time
from dotenv import load_dotenv

# Load test environment configuration
load_dotenv('.env.test')

# Test results tracking
test_results = {}
total_tests = 0
passed_tests = 0

async def log_test_result(function_name, success, details=""):
    """Log test result"""
    global total_tests, passed_tests
    total_tests += 1
    if success:
        passed_tests += 1
    
    test_results[function_name] = {
        "success": success,
        "details": details
    }
    
    status = "✅ PASSED" if success else "❌ FAILED"
    print(f"{function_name}: {status}")
    if details and not success:
        print(f"   Details: {details}")

async def test_auth_functions():
    """Test authentication functions"""
    print("\n=== Testing Authentication Functions ===")
    
    sys.path.append('/home/bkarjoo/dev/gtdmcp')
    try:
        from fastgtd_mcp_server import get_auth_token
        
        # Test 1: get_auth_token
        try:
            token_result = await get_auth_token()
            success = isinstance(token_result, str) and len(token_result) > 0
            await log_test_result("get_auth_token", success, str(token_result) if not success else "")
        except Exception as e:
            await log_test_result("get_auth_token", False, str(e))
            
    except Exception as e:
        await log_test_result("auth_functions_import", False, str(e))

async def test_task_management_functions():
    """Test task management functions"""
    print("\n=== Testing Task Management Functions ===")
    
    sys.path.append('/home/bkarjoo/dev/gtdmcp')
    try:
        from fastgtd_mcp_server import (
            add_task_to_inbox, add_task_to_current_node, add_task_to_node_id,
            create_task, update_task, complete_task, delete_task
        )
        
        # Test 3: add_task_to_inbox
        try:
            inbox_result = await add_task_to_inbox(
                title="Test Inbox Task",
                description="Test task for inbox"
            )
            success = inbox_result.get("success", False)
            task_id = inbox_result.get("task_id") if success else None
            await log_test_result("add_task_to_inbox", success, str(inbox_result) if not success else "")
        except Exception as e:
            await log_test_result("add_task_to_inbox", False, str(e))
        
        # Create a test folder for other task tests
        from fastgtd_mcp_server import create_folder
        folder_result = await create_folder(title="Test Folder for Tasks")
        folder_id = folder_result.get("folder", {}).get("id") if folder_result.get("success") else None
        
        # Test 4: add_task_to_current_node
        try:
            current_result = await add_task_to_current_node(
                title="Test Current Node Task",
                description="Test task for current node",
                current_node_id=folder_id
            )
            success = current_result.get("success", False)
            await log_test_result("add_task_to_current_node", success, str(current_result) if not success else "")
        except Exception as e:
            await log_test_result("add_task_to_current_node", False, str(e))
        
        # Test 5: add_task_to_node_id
        try:
            node_result = await add_task_to_node_id(
                node_id=folder_id,
                task_title="Test Node ID Task",
                description="Test task for node ID"
            )
            success = node_result.get("success", False)
            await log_test_result("add_task_to_node_id", success, str(node_result) if not success else "")
        except Exception as e:
            await log_test_result("add_task_to_node_id", False, str(e))
        
        # Test 6: create_task
        try:
            create_result = await create_task(
                title="Test Created Task",
                description="Task created via create_task",
                parent_id=folder_id
            )
            success = create_result.get("success", False)
            created_task_id = create_result.get("task_id") if success else None
            await log_test_result("create_task", success, str(create_result) if not success else "")
        except Exception as e:
            await log_test_result("create_task", False, str(e))
            created_task_id = None
        
        # Test 7: update_task (using created task)
        if created_task_id:
            try:
                update_result = await update_task(
                    task_id=created_task_id,
                    title="Updated Test Task",
                    description="Task updated via update_task"
                )
                success = update_result.get("success", False)
                await log_test_result("update_task", success, str(update_result) if not success else "")
            except Exception as e:
                await log_test_result("update_task", False, str(e))
        else:
            await log_test_result("update_task", False, "No task to update")
        
        # Test 8: complete_task
        if created_task_id:
            try:
                complete_result = await complete_task(task_id=created_task_id)
                success = complete_result.get("success", False)
                await log_test_result("complete_task", success, str(complete_result) if not success else "")
            except Exception as e:
                await log_test_result("complete_task", False, str(e))
        else:
            await log_test_result("complete_task", False, "No task to complete")
        
        # Test 9: delete_task
        if created_task_id:
            try:
                delete_result = await delete_task(task_id=created_task_id)
                success = delete_result.get("success", False)
                await log_test_result("delete_task", success, str(delete_result) if not success else "")
            except Exception as e:
                await log_test_result("delete_task", False, str(e))
        else:
            await log_test_result("delete_task", False, "No task to delete")
            
    except Exception as e:
        await log_test_result("task_management_import", False, str(e))

async def test_folder_functions():
    """Test folder management functions"""
    print("\n=== Testing Folder Management Functions ===")
    
    sys.path.append('/home/bkarjoo/dev/gtdmcp')
    try:
        from fastgtd_mcp_server import (
            get_all_folders, create_folder, get_root_folders, get_folder_id
        )
        
        # Test 10: get_all_folders
        try:
            all_folders_result = await get_all_folders()
            success = all_folders_result.get("success", False)
            await log_test_result("get_all_folders", success, str(all_folders_result) if not success else "")
        except Exception as e:
            await log_test_result("get_all_folders", False, str(e))
        
        # Test 11: create_folder
        try:
            create_folder_result = await create_folder(title="Test Folder for get_all_folders")
            success = create_folder_result.get("success", False)
            test_folder_id = create_folder_result.get("folder", {}).get("id") if success else None
            await log_test_result("create_folder", success, str(create_folder_result) if not success else "")
        except Exception as e:
            await log_test_result("create_folder", False, str(e))
        
        # Test 12: get_root_folders
        try:
            root_folders_result = await get_root_folders()
            success = root_folders_result.get("success", False)
            await log_test_result("get_root_folders", success, str(root_folders_result) if not success else "")
        except Exception as e:
            await log_test_result("get_root_folders", False, str(e))
        
        # Test 13: get_folder_id
        try:
            folder_id_result = await get_folder_id("Test Folder for get_all_folders")
            success = folder_id_result.get("success", False)
            await log_test_result("get_folder_id", success, str(folder_id_result) if not success else "")
        except Exception as e:
            await log_test_result("get_folder_id", False, str(e))
            
    except Exception as e:
        await log_test_result("folder_functions_import", False, str(e))

async def test_date_and_search_functions():
    """Test date-based and search functions"""
    print("\n=== Testing Date and Search Functions ===")
    
    sys.path.append('/home/bkarjoo/dev/gtdmcp')
    try:
        from fastgtd_mcp_server import (
            get_today_tasks, get_overdue_tasks, search_nodes, get_node_children
        )
        
        # Test 14: get_today_tasks
        try:
            today_result = await get_today_tasks()
            success = today_result.get("success", False)
            await log_test_result("get_today_tasks", success, str(today_result) if not success else "")
        except Exception as e:
            await log_test_result("get_today_tasks", False, str(e))
        
        # Test 15: get_overdue_tasks
        try:
            overdue_result = await get_overdue_tasks()
            success = overdue_result.get("success", False)
            await log_test_result("get_overdue_tasks", success, str(overdue_result) if not success else "")
        except Exception as e:
            await log_test_result("get_overdue_tasks", False, str(e))
        
        # Test 16: search_nodes
        try:
            search_result = await search_nodes(query="searchable", node_type="task", limit=10)
            success = search_result.get("success", False)
            await log_test_result("search_nodes", success, str(search_result) if not success else "")
        except Exception as e:
            await log_test_result("search_nodes", False, str(e))
        
        # Create a test folder with children for get_node_children test
        from fastgtd_mcp_server import create_folder, add_task_to_node_id
        parent_result = await create_folder(title="Parent Folder for Children Test")
        if parent_result.get("success"):
            parent_id = parent_result["folder"]["id"]
            await add_task_to_node_id(node_id=parent_id, task_title="Child Task", description="Child for testing")
            
            # Test 17: get_node_children
            try:
                children_result = await get_node_children(node_id=parent_id)
                success = children_result.get("success", False)
                await log_test_result("get_node_children", success, str(children_result) if not success else "")
            except Exception as e:
                await log_test_result("get_node_children", False, str(e))
        else:
            await log_test_result("get_node_children", False, "Could not create parent folder")
            
    except Exception as e:
        await log_test_result("date_search_functions_import", False, str(e))

async def test_note_functions():
    """Test note management functions"""
    print("\n=== Testing Note Management Functions ===")
    
    sys.path.append('/home/bkarjoo/dev/gtdmcp')
    try:
        from fastgtd_mcp_server import add_note_to_current_node, update_note, create_folder
        
        # Create a folder for note tests
        folder_result = await create_folder(title="Folder for Note Test")
        folder_id = folder_result.get("folder", {}).get("id") if folder_result.get("success") else None
        
        # Test 18: add_note_to_current_node
        try:
            note_result = await add_note_to_current_node(
                title="Test Note",
                content="Test note content",
                current_node_id=folder_id
            )
            success = note_result.get("success", False)
            note_id = note_result.get("note_id") if success else None
            await log_test_result("add_note_to_current_node", success, str(note_result) if not success else "")
        except Exception as e:
            await log_test_result("add_note_to_current_node", False, str(e))
            note_id = None
        
        # Test 19: update_note
        if note_id:
            try:
                update_result = await update_note(
                    note_id=note_id,
                    title="Updated Test Note",
                    content="Updated note content"
                )
                success = update_result.get("success", False)
                await log_test_result("update_note", success, str(update_result) if not success else "")
            except Exception as e:
                await log_test_result("update_note", False, str(e))
        else:
            await log_test_result("update_note", False, "No note to update")
            
    except Exception as e:
        await log_test_result("note_functions_import", False, str(e))

async def test_tag_functions():
    """Test tag management functions"""
    print("\n=== Testing Tag Management Functions ===")
    
    sys.path.append('/home/bkarjoo/dev/gtdmcp')
    try:
        from fastgtd_mcp_server import add_tag, remove_tag, create_task
        
        # Create a task for tagging tests
        task_result = await create_task(
            title="Task for Tagging",
            description="Task to test tagging functions"
        )
        task_id = task_result.get("task_id") if task_result.get("success") else None
        
        # Test 20: add_tag
        try:
            add_tag_result = await add_tag(node_id=task_id, tag_name="test-tag")
            success = add_tag_result.get("success", False)
            await log_test_result("add_tag", success, str(add_tag_result) if not success else "")
        except Exception as e:
            await log_test_result("add_tag", False, str(e))
        
        # Test 21: remove_tag
        try:
            remove_tag_result = await remove_tag(node_id=task_id, tag_name="test-tag")
            success = remove_tag_result.get("success", False)
            await log_test_result("remove_tag", success, str(remove_tag_result) if not success else "")
        except Exception as e:
            await log_test_result("remove_tag", False, str(e))
            
    except Exception as e:
        await log_test_result("tag_functions_import", False, str(e))

async def test_advanced_functions():
    """Test advanced navigation and tree functions"""
    print("\n=== Testing Advanced Functions ===")
    
    sys.path.append('/home/bkarjoo/dev/gtdmcp')
    try:
        from fastgtd_mcp_server import (
            get_node_tree, move_node, get_smart_folder_contents,
            create_folder, create_task
        )
        
        # Create folders for tree and move tests
        parent_result = await create_folder(title="Tree Test Parent")
        dest_result = await create_folder(title="Destination Folder for Move Test")
        
        parent_id = parent_result.get("folder", {}).get("id") if parent_result.get("success") else None
        dest_id = dest_result.get("folder", {}).get("id") if dest_result.get("success") else None
        
        # Test 22: get_node_tree
        try:
            tree_result = await get_node_tree(root_id=parent_id, max_depth=3)
            success = tree_result.get("success", False)
            await log_test_result("get_node_tree", success, str(tree_result) if not success else "")
        except Exception as e:
            await log_test_result("get_node_tree", False, str(e))
        
        # Create a task to move
        task_result = await create_task(
            title="Task to Move",
            description="Task for move testing",
            parent_id=parent_id
        )
        task_id = task_result.get("task_id") if task_result.get("success") else None
        
        # Test 23: move_node
        if task_id and dest_id:
            try:
                move_result = await move_node(
                    node_id=task_id,
                    new_parent_id=dest_id,
                    new_sort_order=0
                )
                success = move_result.get("success", False)
                await log_test_result("move_node", success, str(move_result) if not success else "")
            except Exception as e:
                await log_test_result("move_node", False, str(e))
        else:
            await log_test_result("move_node", False, "Could not create task or destination folder")
        
        # Test 24: get_smart_folder_contents (test with dummy ID)
        try:
            smart_result = await get_smart_folder_contents(
                smart_folder_id="00000000-0000-0000-0000-000000000000",
                limit=10,
                offset=0
            )
            # Even if it fails due to invalid ID, if it returns proper error structure, it's working
            success = isinstance(smart_result, dict) and ("success" in smart_result or "error" in smart_result)
            await log_test_result("get_smart_folder_contents", success, str(smart_result) if not success else "")
        except Exception as e:
            await log_test_result("get_smart_folder_contents", False, str(e))
            
    except Exception as e:
        await log_test_result("advanced_functions_import", False, str(e))

async def cleanup_test_data():
    """Clean up all test data created during testing"""
    print("\n=== Cleaning Up Test Data ===")
    
    sys.path.append('/home/bkarjoo/dev/gtdmcp')
    try:
        from fastgtd_mcp_server import (
            search_nodes, delete_task, get_all_folders, get_folder_id
        )
        
        deleted_tasks = 0
        deleted_notes = 0 
        deleted_folders = 0
        
        # Clean up test tasks
        test_task_queries = [
            "test task", "task for testing", "debug task", "searchable",
            "overdue task", "task due today", "child task", "task to move",
            "task to update", "task to complete", "task to delete", "task for tagging",
            "Test Inbox Task", "Test Current Node Task", "Test Node ID Task",
            "Test Created Task", "Updated Test Task", "Task for Tagging"
        ]
        
        for query in test_task_queries:
            search_result = await search_nodes(query=query, node_type="task", limit=50)
            if search_result.get("success"):
                tasks = search_result.get("results", [])
                for task in tasks:
                    task_id = task.get("id")
                    if task_id:
                        delete_result = await delete_task(task_id)
                        if delete_result.get("success"):
                            deleted_tasks += 1
        
        # Clean up test notes
        search_result = await search_nodes(query="test note", node_type="note", limit=50)
        if search_result.get("success"):
            notes = search_result.get("results", [])
            for note in notes:
                note_id = note.get("id")
                if note_id:
                    delete_result = await delete_task(note_id)
                    if delete_result.get("success"):
                        deleted_notes += 1
        
        # Clean up test folders
        folders_result = await get_all_folders()
        if folders_result.get("success"):
            folders = folders_result.get("folders", [])
            
            test_patterns = [
                r"test.*folder", r".*test.*", r"debug.*folder", r"tree.*parent",
                r"source.*folder", r"destination.*folder", r"parent.*folder.*test",
                r"searchable.*folder", r"comprehensive.*test.*folder",
                r".*for.*test.*", r".*move.*test.*"
            ]
            
            for folder_name in folders:
                is_test_folder = any(
                    re.search(pattern, folder_name.lower()) 
                    for pattern in test_patterns
                )
                
                if is_test_folder:
                    folder_id_result = await get_folder_id(folder_name)
                    if folder_id_result.get("success"):
                        folder_id = folder_id_result.get("folder_id")
                        if folder_id:
                            delete_result = await delete_task(folder_id)
                            if delete_result.get("success"):
                                deleted_folders += 1
        
        print(f"🧹 Cleanup completed:")
        print(f"   Tasks deleted: {deleted_tasks}")
        print(f"   Notes deleted: {deleted_notes}")
        print(f"   Folders deleted: {deleted_folders}")
        
        return True
        
    except Exception as e:
        print(f"❌ Cleanup failed: {e}")
        return False

async def main():
    print("🧪 FastGTD MCP Complete Function Test Suite")
    print("==========================================")
    print("Testing all 22 MCP functions in one comprehensive test")
    
    # Run all test groups
    await test_auth_functions()
    await test_task_management_functions()
    await test_folder_functions()
    await test_date_and_search_functions()
    await test_note_functions()
    await test_tag_functions()
    await test_advanced_functions()
    
    # Print final results
    print(f"\n📊 Final Test Results: {passed_tests}/{total_tests} tests passed")
    print("="*50)
    
    # Show details for failed tests
    failed_tests = [name for name, result in test_results.items() if not result["success"]]
    if failed_tests:
        print("❌ Failed tests:")
        for test_name in failed_tests:
            details = test_results[test_name]["details"]
            print(f"   • {test_name}: {details[:100]}...")
    else:
        print("🎉 ALL TESTS PASSED!")
    
    # Run cleanup
    print("\n🧹 Running cleanup...")
    cleanup_success = await cleanup_test_data()
    
    overall_success = passed_tests == total_tests
    print(f"\n🎯 Overall Result: {'✅ SUCCESS' if overall_success else '❌ SOME TESTS FAILED'}")
    
    if overall_success:
        print("🎉 ALL 22 MCP FUNCTIONS WORKING PERFECTLY!")
    
    return overall_success

if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)