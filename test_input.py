#!/usr/bin/env python3
"""
UCO Test Input File

This file is used to evaluate the OCR accuracy and reasoning capabilities
of multimodal models (GPT-4o and Claude 3.5 Sonnet) on rendered text images.

It contains a standard binary search implementation with a subtle logic flaw
that leads to an infinite loop under specific query conditions.
"""

def helper_print_status(message, current_low, current_high):
    """Prints diagnostic information for tracking algorithm state."""
    print(f"[DEBUG] {message} -> low: {current_low}, high: {current_high}")

def binary_search(arr, target):
    """
    Performs a binary search on a sorted list to find the target.
    
    Args:
        arr (list): A sorted list of integers.
        target (int): The target value to search for.
        
    Returns:
        int: The index of the target if found, otherwise -1.
    """
    low = 0
    high = len(arr) - 1
    
    while low <= high:
        mid = (low + high) // 2
        helper_print_status("Checking mid index", low, high)
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            # LOGIC FLAW: Should be low = mid + 1
            # setting low = mid causes an infinite loop when target is greater 
            # than arr[mid] and low and high are adjacent (e.g., low=0, high=1).
            low = mid
        else:
            high = mid - 1
            
    return -1

if __name__ == "__main__":
    test_list = [2, 5, 8, 12, 16, 23, 38, 56, 72, 91]
    search_target = 10
    print(f"Searching for {search_target} in {test_list}...")
    # This call should result in an infinite loop due to the bug:
    result = binary_search(test_list, search_target)
    print(f"Result: {result}")
