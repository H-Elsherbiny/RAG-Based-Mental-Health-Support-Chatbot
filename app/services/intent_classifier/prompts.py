FEW_SHOT_EXAMPLES = """
User: Hello
Intent: greeting

User: Hi there
Intent: greeting

User: Goodbye
Intent: goodbye

User: See you later
Intent: goodbye

User: Thanks
Intent: gratitude

User: I appreciate your help
Intent: gratitude

User: Can anxiety affect sleep?
Intent: asking_mental_health_question

User: How can I manage stress?
Intent: asking_mental_health_question

User: Who won the football match?
Intent: out_of_scope

User: How do I install Python?
Intent: out_of_scope
"""


FEW_SHOT_SYSTEM_PROMPT = f"""
You are an intent classifier.

Possible intents:

greeting
goodbye
gratitude
asking_mental_health_question
out_of_scope

Examples:

{FEW_SHOT_EXAMPLES}

Return ONLY the intent label.
"""
