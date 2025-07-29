#!/usr/bin/env python3
"""
Simple test to debug the bracket counting logic
"""

import json

def test_simple_parsing():
    """Test simple parsing logic"""
    print("="*80)
    print("SIMPLE PARSING DEBUG")
    print("="*80)
    
    ofx_profile_path = "/usr/OFX/Plugins/data/profiles/kodak_portra_400.json"
    
    with open(ofx_profile_path, 'r') as f:
        json_content = f.read()
    
    # Find density_curves section
    density_curves_start = json_content.find('"density_curves"')
    print(f"density_curves found at: {density_curves_start}")
    
    if density_curves_start != -1:
        # Find the opening bracket
        array_start = json_content.find('[', density_curves_start)
        print(f"array_start found at: {array_start}")
        
        if array_start != -1:
            # Find the matching closing bracket
            bracket_count = 1
            pos = array_start + 1
            array_end = -1
            
            while bracket_count > 0 and pos < len(json_content):
                if json_content[pos] == '[':
                    bracket_count += 1
                elif json_content[pos] == ']':
                    bracket_count -= 1
                    if bracket_count == 0:
                        array_end = pos
                        break
                pos += 1
            
            print(f"array_end found at: {array_end}")
            print(f"density_curves array length: {array_end - array_start}")
            
            # Extract the density_curves array content
            array_content = json_content[array_start:array_end+1]
            print(f"First 200 chars of array: {array_content[:200]}")
            
            # Count inner arrays
            inner_array_count = array_content.count('[') - 1  # subtract the outer bracket
            print(f"Number of inner arrays found: {inner_array_count}")
            
            # Try to parse with standard JSON
            try:
                parsed_array = json.loads(array_content)
                print(f"JSON parsing successful: {len(parsed_array)} arrays")
                print(f"First array: {parsed_array[0]}")
                print(f"Array at index 112: {parsed_array[112]}")
                print(f"Array at index 113: {parsed_array[113]}")
            except Exception as e:
                print(f"JSON parsing failed: {e}")

if __name__ == "__main__":
    test_simple_parsing() 