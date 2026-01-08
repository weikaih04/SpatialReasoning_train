import re

def test_extraction():
    examples = [
        "In this scene, after reaching the 'X' position on the ground (keeping the same facing direction) and then performing a 90-degree right turn, would the doorway be at a closer or further distance?",
        "Suppose you move to the 'X' mark on the ground without changing your facing direction, then turn right by 90 degrees. Will the stool be closer or further from you?",
        "Suppose you move to the 'X' mark on the ground without changing your facing direction, then turn left by 90 degrees. Will the garbagecan be on your left or right?",
        "Say I walk to the 'X' spot on the ground (facing the same direction as now) and then turn left 90 degrees - will the sidetable be on my left or my right?",
        "For someone who moves straight to the 'X' marked point on the ground while maintaining their orientation, then turns right for 90 degrees, will the spraybottle be on their left or right?",
        "Upon moving to the marked 'X' on the ground (keeping the same facing direction) and then turning right for 90 degrees, would you find the chair on the right on your left or right?"
    ]

    for question_text in examples:
        print(f"\n--- Testing: {question_text[:50]}... ---")
        
        # Split logic
        parts = re.split(r'[.?!]', question_text)
        parts = [p.strip() for p in parts if p.strip()]
        last_sentence = parts[-1] if parts else question_text
        print(f"Last Sentence: '{last_sentence}'")

        # Direction Logic
        direction = "specified direction"
        # ORIGINAL BROKEN: r"turn(?:ing)?\s+(left|right)"
        # FIX ATTEMPT: Look for "turn" word stem and subsequent direction
        # Handle "right turn", "turns right", "turn left"
        
        # 1. "turn... (left/right)"
        match_dir_1 = re.search(r"turn[s|ing]*\s+.*?(left|right)", question_text, re.IGNORECASE)
        # 2. "(left/right) turn"
        match_dir_2 = re.search(r"(left|right)\s+turn", question_text, re.IGNORECASE)
        
        if match_dir_1:
             direction = match_dir_1.group(1).lower()
        elif match_dir_2:
             direction = match_dir_2.group(1).lower()
             
        print(f"Direction: {direction}")

        # Object Logic
        obj = "target object"
        
        # FIX ATTEMPT: Simpler regex, split by common verbs?
        # Pattern 1: "would/will the [Noun] be"
        # Pattern 2: "find the [Noun] on"
        
        # Use simpler non-greedy search
        match_obj = re.search(r"(?:would|will|find)\s+(?:the|a)\s+(?P<obj>\w+(?:\s+\w+){0,3}?)\s+(?:be|on)", last_sentence, re.IGNORECASE)
        
        if match_obj:
            obj = match_obj.group("obj")
        else:
            # Fallback for "will the sidetable be on..."
            # Maybe my regex was incorrectly requiring space-be-space?
            # Let's clean the sentence first?
            pass

        print(f"Object: {obj}")

test_extraction()
