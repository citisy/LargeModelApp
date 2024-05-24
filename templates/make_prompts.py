from typing import List, Callable
import os

lang = os.environ.get('LANG', 'en')

if lang == 'ch':
    from .ch import *
else:
    from .en import *


def make_role(role):
    return role_str.format(role=role)


def make_today_date():
    import datetime
    today_date = datetime.datetime.now().strftime('%Y-%m-%d')
    return today_date_str.format(today_date=today_date)


def make_location(location: str):
    return location_str.format(location=location)


def make_access_tools(access_tools: dict):
    """access_tools = {'name': {'func': ..., 'desc': ..., 'args': ...}}"""
    tmp = []
    for name, dic in access_tools.items():
        desc = dic.get('desc', dic['func'].__doc__)
        args = dic['args']
        _args = ''
        if isinstance(args, list):
            # args = [{'name': ..., 'description': ..., 'type', ...}]
            for a in args:
                _args += '{' + access_tool_args_unit_str.format(**a) + '}'
            args = '[' + ''.join(_args) + ']'
        access_tool_unit_str.format(tool_name=name, desc=desc, args=args)
        tmp.append(access_tool_unit_str.format(tool_name=name, desc=desc, args=args))

    access_tools = '\n'.join(tmp)
    return access_tools_str.format(access_tools=access_tools)


def make_chat_history(chat_history: List[List[str]]):
    tmp = []
    for q, a in chat_history:
        tmp.append(chat_history_unit_str.format(question=q, answer=a))
    chat_history = '\n'.join(tmp)
    return chat_history_str.format(chat_history=chat_history)


def make_thought_chain(tool_names: List[str]):
    return thought_chain_str.format(tool_names=', '.join(tool_names))


def make_thought_chain_query(query, thought, action, action_input, observation):
    return thought_chain_query_str.format(
        query=query,
        thought=thought,
        action=action,
        action_input=action_input,
        observation=observation
    )


def make_query(query):
    return query_str.format(query=query)


def make_encourage():
    return encourage_best_str


def make_skeleton(query: str = '', additional_prompts: List[str] = []):
    additional_prompt = ''.join(additional_prompts)
    return skeleton_str.format(additional_prompt=additional_prompt, query_str=query)
