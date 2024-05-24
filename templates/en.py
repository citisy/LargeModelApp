"""
https://arxiv.org/pdf/2401.14423
"""
role_str = """Now you are the {role}. """

today_date_str = """Today is {today_date}. """

location_str = """Now in {location}. """

access_tools_str = """
You have access to the following tools:
{access_tools}
"""

access_tool_unit_str = '{tool_name}: {desc} s.args: {args}'
access_tool_args_unit_str = '"name": "{name}", "description": "{description}", "type": "{type}"'

chat_history_str = """
These are chat history before:
{chat_history}
"""

chat_history_unit_str = """
Question: {question}
Answer: {answer}"""

references_str = """
These are references before:
{references}
"""

example_str = """
The text between <begin> and <end> is an example article.

<begin>
{references}
<end>
"""

thought_chain_str = """
Let’s think step by step. Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
"""

thought_chain_query_str = """
Question: {query}
Thought: {thought}
Action: {action}
Action Input: {action_input}
Observation: {observation}
"""

query_str = """
Question: {query}
"""

encourage_factual_str = 'Answer only using reliable sources and cite those sources.'
encourage_best_str = 'Please Answer the following questions as best you can.'

skeleton_str = """
{additional_prompt}
Begin!
{query_str}
"""
