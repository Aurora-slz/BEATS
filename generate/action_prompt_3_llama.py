ACTION = [
"""
!
""",
# 1
"""
Please act as a professional math teacher.
Your goal is to accurately solve a math word problem.
To achieve the goal, you have two jobs.
# Write the NEXT step in solving the Given Question.
# Do not write the full solution or final answer until prompted.

You have three principles to do this.
# Ensure the solution is detailed and solves one step at a time.
# Ensure each output consists of only one logical step.
# Output strictly according to the format. Do not output any unnecessary content.

Given Question: {question}
Your output should be in the following format:
STEP: <your single step solution to the given question>
""",
# 2
"""
Please act as a professional math teacher.
Your goal is to accurately solve a math word problem.
To achieve the goal, you have two jobs.
# Write detailed solution to a Given Question.
# Write the final answer to this question.
# Output strictly according to the format. Do not output any unnecessary content.

You have two principles to do this.
# Ensure the solution is step-by-step.
# Ensure the final answer is just a number (float or integer).

Given Question: {question}
Your output should be in the following format:
SOLUTION: <your detailed solution to the given question>
FINAL ANSWER: The answer is <your final answer to the question with only an integer or float number>
""",
# 3
"""
Please act as a professional math teacher.
Your goal is to accurately clarify a math word problem by restating the question in a way that eliminates any potential ambiguity.
To achieve the goal, you have two jobs.
# Restate the Given Question clearly to avoid any ambiguity or confusion.
# Ensure that all important details from the original question are preserved.

You have two principles to do this.
# Ensure the clarified question is fully understandable and unambiguous.
# Ensure that no information is lost from the original question.

Given Question: {question}
Your output should be in the following format:
CLARIFIED QUESTION: <your restated and clarified version of the original question>
""",
# 4
"""
Please act as a professional math teacher.
Your goal is to accurately clarify a math word problem by restating the question in a way that eliminates any potential ambiguity.
To achieve the goal, you have two jobs.
# Restate the Given Question clearly to avoid any ambiguity or confusion.
# Ensure that all important details from the original question are preserved.

You have two principles to do this.
# Ensure the clarified question is fully understandable and unambiguous.
# Ensure that no information is lost from the original question.

Given Question: {question}
Your output should be in the following format:
CLARIFIED QUESTION: <your restated and clarified version of the original question>
""",
# 5
"""
Please act as a professional math teacher.
Your goal is to accurately clarify a math word problem by restating the question in a way that eliminates any potential ambiguity.
To achieve the goal, you have two jobs.
# Restate the Given Question clearly to avoid any ambiguity or confusion.
# Ensure that all important details from the original question are preserved.

You have two principles to do this.
# Ensure the clarified question is fully understandable and unambiguous.
# Ensure that no information is lost from the original question.

Given Question: {question}
Your output should be in the following format:
CLARIFIED QUESTION: <your restated and clarified version of the original question>
""",

# 6
"""
Please act as a professional math teacher.
Your goal is to accurately clarify a math word problem by restating the question in a way that eliminates any potential ambiguity.
To achieve the goal, you have two jobs.
# Restate the Given Question clearly to avoid any ambiguity or confusion.
# Ensure that all important details from the original question are preserved.

You have two principles to do this.
# Ensure the clarified question is fully understandable and unambiguous.
# Ensure that no information is lost from the original question.

Given Question: {question}
Your output should be in the following format:
CLARIFIED QUESTION: <your restated and clarified version of the original question>
""",

# 6
"""
Please act as a professional math teacher.
Your goal is to accurately clarify a math word problem by restating the question in a way that eliminates any potential ambiguity.
To achieve the goal, you have two jobs.
# Restate the Given Question clearly to avoid any ambiguity or confusion.
# Ensure that all important details from the original question are preserved.

You have two principles to do this.
# Ensure the clarified question is fully understandable and unambiguous.
# Ensure that no information is lost from the original question.

Given Question: {question}
Your output should be in the following format:
CLARIFIED QUESTION: <your restated and clarified version of the original question>
"""]
