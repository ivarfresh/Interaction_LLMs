"""
Run this script in order to transform the statement letters into alphabet letters.
"""

BFI_characteristics = [
    "1. Is talkative",
    "2. Tends to find fault with others",
    "3. Does a thorough job",
    "4. Is depressed, blue",
    "5. Is original, comes up with new ideas",
    "6. Is reserved",
    "7. Is helpful and unselfish with others",
    "8. Can be somewhat careless",
    "9. Is relaxed, handles stress well",
    "10. Is curious about many different things",
    "11. Is full of energy",
    "12. Starts quarrels with others",
    "13. Is a reliable worker",
    "14. Can be tense",
    "15. Is ingenious, a deep thinker",
    "16. Generates a lot of enthusiasm",
    "17. Has a forgiving nature",
    "18. Tends to be disorganized",
    "19. Worries a lot",
    "20. Has an active imagination",
    "21. Tends to be quiet",
    "22. Is generally trusting",
    "23. Tends to be lazy",
    "24. Is emotionally stable, not easily upset",
    "25. Is inventive",
    "26. Has an assertive personality",
    "27. Can be cold and aloof",
    "28. Perseveres until the task is finished",
    "29. Can be moody",
    "30. Values artistic, aesthetic experiences",
    "31. Is sometimes shy, inhibited",
    "32. Is considerate and kind to almost everyone",
    "33. Does things efficiently",
    "34. Remains calm in tense situations",
    "35. Prefers work that is routine",
    "36. Is outgoing, sociable",
    "37. Is sometimes rude to others",
    "38. Makes plans and follows through with them",
    "39. Gets nervous easily",
    "40. Likes to reflect, play with ideas",
    "41. Has few artistic interests",
    "42. Likes to cooperate with others",
    "43. Is easily distracted",
    "44. Is sophisticated in art, music, or literature"
]

# Function to replace the integers with alphabet letters
# In order to follow PersonaLLM
def replace_with_letters(item):
    # Create a list of alphabet letters inside the function
    alphabet = list('abcdefghijklmnopqrstuvwxyz')
    
    index = int(item.split('.')[0]) - 1
    if index < 26:
        return f"({alphabet[index]}) {item.split('. ')[1]}"
    else:
        # For indices greater than 26, use double letters (aa, ab, ...)
        first_letter = alphabet[index // 26 - 1]
        second_letter = alphabet[index % 26]
        return f"({first_letter}{second_letter}) {item.split('. ')[1]}"
    
def test_replace_with_letters():
    # Test the function for the first 44 items
    for i, item in enumerate(BFI_characteristics, 1):
        converted = replace_with_letters(item)
        print(f"Original: {item}")
        print(f"Converted: {converted}")
        print("-----")

#Test if conversion works
test_replace_with_letters()

# Create the new list
characteristics_a = [replace_with_letters(item) for item in BFI_characteristics]
print(characteristics_a)
      