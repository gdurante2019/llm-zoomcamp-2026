"""
Utility module for displaying truncated output in Jupyter notebooks.

Provides the show() function for clean display of lists and dictionaries
with truncated values to prevent output clutter.
"""

import re


def show(obj, max_items=20, max_chars_per_item=200):
    """
    Display object with truncation and clean formatting.
    
    Handles:
    - Lists of dictionaries: displays each dict on new line with key-value pairs
    - Regular lists: displays each item on new line
    - Single dictionaries: displays key-value pairs
    
    Features:
    - Truncates long values to max_chars_per_item
    - Removes problematic whitespace (newlines, tabs)
    - Maintains readable formatting for nested structures
    
    Args:
        obj: Object to display (list, dict, or list of dicts)
        max_items: Maximum number of items to show (default: 20)
        max_chars_per_item: Maximum characters per value (default: 200)
    
    Examples:
        >>> documents = [
        ...     {'content': 'very long text...', 'filename': 'file1.md'},
        ...     {'content': 'more text...', 'filename': 'file2.md'}
        ... ]
        >>> show(documents)
        
        >>> my_list = ['long string item 1', 'long string item 2', 'item 3']
        >>> show(my_list, max_chars_per_item=100)
    """
    
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        # List of dictionaries
        output = "["
        for i, item in enumerate(obj[:max_items]):
            if i > 0:
                output += "\n"
            output += "{"
            keys = list(item.keys())
            for j, key in enumerate(keys):
                value = item[key]
                key_str = str(key)
                value_str = str(value)
                
                # Remove newlines, tabs, carriage returns
                value_str = re.sub(r'[\n\r\t]+', '', value_str)
                
                # Truncate if needed
                if len(key_str) > max_chars_per_item:
                    key_str = key_str[:max_chars_per_item] + "..."
                if len(value_str) > max_chars_per_item:
                    value_str = value_str[:max_chars_per_item] + "..."
                
                comma = "," if j < len(keys) - 1 else ""
                if j == 0:
                    output += f"'{key_str}': '{value_str}'{comma}"
                else:
                    output += f"\n  '{key_str}': '{value_str}'{comma}"
            
            comma = "," if i < len(obj[:max_items]) - 1 else ""
            output += f"}}{comma}"
        
        output += "]"
        print(output)
        if len(obj) > max_items:
            print(f"(and {len(obj) - max_items} more items)")
    
    elif isinstance(obj, list):
        # Regular list of strings/items
        output = "["
        for i, item in enumerate(obj[:max_items]):
            if i > 0:
                output += "\n "
            item_str = re.sub(r'[\n\r\t]+', '', str(item))
            if len(item_str) > max_chars_per_item:
                item_str = item_str[:max_chars_per_item] + "..."
            comma = "," if i < len(obj[:max_items]) - 1 else ""
            output += f"'{item_str}'{comma}"
        
        output += "]"
        print(output)
        if len(obj) > max_items:
            print(f"(and {len(obj) - max_items} more items)")
    
    elif isinstance(obj, dict):
        # Single dictionary
        output = "{"
        keys = list(obj.keys())
        for j, key in enumerate(keys):
            value = obj[key]
            key_str = str(key)
            value_str = str(value)
            
            value_str = re.sub(r'[\n\r\t]+', '', value_str)
            
            if len(key_str) > max_chars_per_item:
                key_str = key_str[:max_chars_per_item] + "..."
            if len(value_str) > max_chars_per_item:
                value_str = value_str[:max_chars_per_item] + "..."
            
            comma = "," if j < len(keys) - 1 else ""
            if j == 0:
                output += f"'{key_str}': '{value_str}'{comma}"
            else:
                output += f"\n  '{key_str}': '{value_str}'{comma}"
        
        output += "}"
        print(output)
