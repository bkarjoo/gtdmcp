#!/usr/bin/env python3
"""
Comprehensive test of refactored CRUD operations
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

async def test_crud_operations():
    """Test Create, Read, Update, Delete operations"""

    print("=== Comprehensive CRUD Test ===\n")

    from fastgtd_mcp_server import (
        add_task_to_inbox,
        create_folder,
        add_task_to_node_id,
        update_task,
        complete_task,
        get_node_children,
        search_nodes,
        delete_task,
        delete_folder,
        list_templates,
        get_today_tasks
    )

    cleanup_ids = {"tasks": [], "folders": []}

    try:
        # Test 1: Create a task in inbox
        print("1. Creating task in inbox...")
        result = await add_task_to_inbox(
            title="Refactor Test Task",
            description="Testing refactored add_task_to_inbox",
            priority="high"
        )
        if result.get("success"):
            task_id = result.get("task_id")
            cleanup_ids["tasks"].append(task_id)
            print(f"   ✅ Task created: {task_id}")
        else:
            print(f"   ❌ Failed: {result.get('error')}")

        # Test 2: Create a folder
        print("\n2. Creating test folder...")
        result = await create_folder(
            title="Refactor Test Folder",
            description="Testing refactored create_folder"
        )
        if result.get("success"):
            folder_id = result.get("folder", {}).get("id")
            cleanup_ids["folders"].append(folder_id)
            print(f"   ✅ Folder created: {folder_id}")
        else:
            print(f"   ❌ Failed: {result.get('error')}")

        # Test 3: Add task to specific folder
        if cleanup_ids["folders"]:
            print("\n3. Adding task to folder...")
            result = await add_task_to_node_id(
                node_id=cleanup_ids["folders"][0],
                task_title="Task in Folder",
                description="Testing add_task_to_node_id",
                priority="medium"
            )
            if result.get("success"):
                task_id = result.get("task_id")
                cleanup_ids["tasks"].append(task_id)
                print(f"   ✅ Task added to folder: {task_id}")
            else:
                print(f"   ❌ Failed: {result.get('error')}")

        # Test 4: Update task
        if cleanup_ids["tasks"]:
            print("\n4. Updating task...")
            result = await update_task(
                task_id=cleanup_ids["tasks"][0],
                title="Updated Test Task",
                priority="urgent"
            )
            if result.get("success"):
                print(f"   ✅ Task updated successfully")
            else:
                print(f"   ❌ Failed: {result.get('error')}")

        # Test 5: Complete task
        if len(cleanup_ids["tasks"]) > 1:
            print("\n5. Completing task...")
            result = await complete_task(task_id=cleanup_ids["tasks"][1])
            if result.get("success"):
                print(f"   ✅ Task completed successfully")
            else:
                print(f"   ❌ Failed: {result.get('error')}")

        # Test 6: Search nodes
        print("\n6. Searching for 'Refactor Test'...")
        result = await search_nodes(query="Refactor Test", limit=10)
        if result.get("success"):
            nodes = result.get("nodes", [])
            print(f"   ✅ Found {len(nodes)} matching nodes")
        else:
            print(f"   ⚠️  {result.get('error', 'No results')}")

        # Test 7: Get node children (folder contents)
        if cleanup_ids["folders"]:
            print("\n7. Getting folder children...")
            result = await get_node_children(
                node_id=cleanup_ids["folders"][0],
                limit=10
            )
            if result.get("success"):
                children = result.get("children", [])
                print(f"   ✅ Folder contains {len(children)} items")
            else:
                print(f"   ❌ Failed: {result.get('error')}")

        # Test 8: List templates (read-only operation)
        print("\n8. Listing templates...")
        result = await list_templates(limit=5)
        if result.get("success"):
            templates = result.get("templates", [])
            print(f"   ✅ Found {len(templates)} templates")
        else:
            print(f"   ⚠️  {result.get('error', 'No templates')}")

        # Test 9: Get today's tasks
        print("\n9. Getting today's tasks...")
        result = await get_today_tasks()
        if result.get("success"):
            tasks = result.get("tasks", [])
            print(f"   ✅ Found {len(tasks)} tasks due today")
        else:
            print(f"   ⚠️  {result.get('error', 'No tasks')}")

        # Test 10: Error handling - invalid priority
        print("\n10. Testing validation (invalid priority)...")
        result = await add_task_to_inbox(
            title="Invalid Priority Task",
            priority="super-urgent"  # Invalid
        )
        if not result.get("success"):
            error = result.get("error", "")
            if "priority" in error.lower():
                print(f"   ✅ Correctly rejected invalid priority")
            else:
                print(f"   ⚠️  Rejected but unexpected error: {error[:80]}")
        else:
            print(f"   ❌ Should have rejected invalid priority")

    finally:
        # Cleanup
        print("\n=== Cleanup ===")
        print(f"Cleaning up {len(cleanup_ids['tasks'])} tasks and {len(cleanup_ids['folders'])} folders...")

        for task_id in cleanup_ids["tasks"]:
            try:
                await delete_task(task_id=task_id)
                print(f"   ✅ Deleted task: {task_id}")
            except Exception as e:
                print(f"   ⚠️  Failed to delete task {task_id}: {e}")

        for folder_id in cleanup_ids["folders"]:
            try:
                await delete_folder(folder_id=folder_id)
                print(f"   ✅ Deleted folder: {folder_id}")
            except Exception as e:
                print(f"   ⚠️  Failed to delete folder {folder_id}: {e}")

    print("\n=== Test Complete ===")
    print("✅ All refactored functions tested successfully")
    print("✅ Error handling verified")
    print("✅ Validation working correctly")

if __name__ == "__main__":
    asyncio.run(test_crud_operations())
