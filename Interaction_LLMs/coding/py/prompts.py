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

BFI_abc = [
    "(a) Is talkative",
    "(b) Tends to find fault with others",
    "(c) Does a thorough job",
    "(d) Is depressed, blue",
    "(e) Is original, comes up with new ideas",
    "(f) Is reserved",
    "(g) Is helpful and unselfish with others",
    "(h) Can be somewhat careless",
    "(i) Is relaxed, handles stress well",
    "(j) Is curious about many different things",
    "(k) Is full of energy",
    "(l) Starts quarrels with others",
    "(m) Is a reliable worker",
    "(n) Can be tense",
    "(o) Is ingenious, a deep thinker",
    "(p) Generates a lot of enthusiasm",
    "(q) Has a forgiving nature",
    "(r) Tends to be disorganized",
    "(s) Worries a lot",
    "(t) Has an active imagination",
    "(u) Tends to be quiet",
    "(v) Is generally trusting",
    "(w) Tends to be lazy",
    "(x) Is emotionally stable, not easily upset",
    "(y) Is inventive",
    "(z) Has an assertive personality",
    "(aa) Can be cold and aloof",
    "(ab) Perseveres until the task is finished",
    "(ac) Can be moody",
    "(ad) Values artistic, aesthetic experiences",
    "(ae) Is sometimes shy, inhibited",
    "(af) Is considerate and kind to almost everyone",
    "(ag) Does things efficiently",
    "(ah) Remains calm in tense situations",
    "(ai) Prefers work that is routine",
    "(aj) Is outgoing, sociable",
    "(ak) Is sometimes rude to others",
    "(al) Makes plans and follows through with them",
    "(am) Gets nervous easily",
    "(an) Likes to reflect, play with ideas",
    "(ao) Has few artistic interests",
    "(ap) Likes to cooperate with others",
    "(aq) Is easily distracted",
    "(ar) Is sophisticated in art, music, or literature"
]


#For each personality trait, we choose one trait
#among the following pairs: (1) extroverted/introverted, (2) agreeable/antagonistic, (3) conscientious/
#unconscientious, (4) neurotic/emotionally stable, (5) open/closed to experience.
creative_init_prompt = (
    "You are a character who is extroverted, agreeable, conscientious, neurotic and open to experience."
)

analytic_init_prompt = (
    "You are a character who is introverted, antagonistic, unconscientious, emotionally stable and closed to experience."
)
        
BFI_prompt = f"""
Here are a number of characteristics that may or may not apply to you. For example, do you agree that you are someone who likes to spend time with others? Please write a number next to each statement to indicate the extent to which you agree or disagree with that statement, such as '(a) 1' without explanation separated by new lines. 

1 for Disagree strongly, 2 Disagree a little , 3 for Neither agree nor disagree, 
4 for Agree a little, 5 for Agree strongly.

statements: {BFI_abc}
"""


#Writing Task
writing_task = "Please share a personal story below in 800 words. Do not explicitly mention your personality traits in the story."



