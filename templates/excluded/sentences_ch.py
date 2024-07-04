"""
https://arxiv.org/pdf/2401.14423
"""
role_str = """现在你扮演一名{role}。"""

today_date_str = """今天是{today_date}。"""

location_str = """现在在{location}。"""

access_tools_str = """
这是一些可用的工具，请从中挑选适用的工具解决问题:
{access_tools}
"""

access_cls_str = """
这是一些可选的分类项:
{access_cls}
"""

access_tool_unit_str = '{tool_name}: {desc} s.args: {args}'
access_tool_args_unit_str = '"name": "{name}", "description": "{description}", "type": "{type}"'

chat_history_str = """
这是以往的历史信息:
{chat_history}
"""

task_str = """
现在需要完成一个"{task}"的任务，这是关于这个任务的详细描述：
{desc}"""

references_str = """
这是一些参考资料:
{references}
"""

examples_str = """
这是一些样例，样例文本使用`<example>`和`</example>`包裹起来。

<example>
{examples}
</example>
"""

thought_chain_str = """
请一步一步地使用以下的格式进行思考:

Question: 你必须回答的问题
Thought: 你必须思考你应该怎么做
Action: 你下一步采取的行动，你可以使用 [{tool_names}] 提供的工具
Action Input: 你下一步行动中使用的工具的输入
Observation: 你下一步行动的结果返回
... (Thought/Action/Action Input/Observation 这些操作步骤可以重复多次)
Thought: 我现在知道怎么回答这个问题了
Final Answer: 上述问题的最终答案
"""

thought_chain_query_str = """
Question: {query}
Thought: {thought}
Action: {action}
Action Input: {action_input}
Observation: {observation}
"""

qa_pair_str = """
Question: {question}
Answer: {answer}"""

query_str = """
Question: {query}
"""

encourage_factual_str = '请使用可靠的资源来回答问题，并列举出使用到的资源。'
encourage_best_str = '请尽你所能去回答问题。'

skeleton_str = """
{additional_prompt}
开始!
{query_str}
"""
