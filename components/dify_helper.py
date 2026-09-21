"""Runtime helpers for converted Dify workflows.

Converted modules read/write node variables via `{{#node_id.var#}}`
and render prompts / templates.
"""
import json
import os
import re
from typing import Any, Optional, Sequence

SELECTOR_RE = re.compile(r'\{\{#([^#]+)#\}\}')
JINJA_VAR_RE = re.compile(r'\{\{\s*([a-zA-Z_][\w]*)\s*\}\}')


def sel(obj: dict, node_id: Any, *keys: str, default: Any = ''):
    """Resolve a Dify variable selector `{{#node_id.key#}}`."""
    node_id = str(node_id)
    data = (obj.get('_dify') or {}).get(node_id)
    if data is None:
        if not keys:
            return obj.get(node_id, default)
        if len(keys) == 1 and keys[0] in obj:
            return obj.get(keys[0], default)
        return default
    cur = data
    for i, key in enumerate(keys):
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        elif i == 0 and key in obj:
            cur = obj[key]
        else:
            return default
    return default if cur is None else cur


def set_node(obj: dict, node_id: Any, **values):
    """Write node outputs to obj['_dify'][node_id] and return obj in place (for MultiThreadPipeline)."""
    node_id = str(node_id)
    obj.setdefault('_dify', {})
    bucket = obj['_dify'].setdefault(node_id, {})
    bucket.update(values)
    return obj


def render_prompt(text: Any, obj: dict) -> str:
    """Replace Dify placeholders `{{#node_id.var#}}` in a prompt string."""
    if text is None:
        return ''
    text = str(text)

    def _selector_value(obj: dict, expr: str):
        parts = [p for p in str(expr).split('.') if p]
        if not parts:
            return ''
        if parts[0] in ('env', 'environment'):
            return os.getenv('.'.join(parts[1:]), '')
        if len(parts) == 1:
            return obj.get(parts[0], sel(obj, parts[0], default=''))
        return sel(obj, parts[0], *parts[1:])

    def repl(match: re.Match) -> str:
        value = _selector_value(obj, match.group(1))
        if value is None:
            return ''
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    return SELECTOR_RE.sub(repl, text)


def render_template(template: Any, obj: dict, variables: Optional[dict] = None) -> str:
    """Render a template-transform node: Dify selectors first, then Jinja / `{{ var }}`."""
    text = render_prompt(template, obj)
    ctx = dict(variables or {})
    if '{%' in text or '{#' in text:
        try:
            from jinja2 import Template
            return Template(text).render(**ctx)
        except Exception:
            pass

    def repl(match: re.Match) -> str:
        key = match.group(1)
        if key in ctx:
            value = ctx[key]
            return '' if value is None else str(value)
        return match.group(0)

    return JINJA_VAR_RE.sub(repl, text)


def parse_http_headers(text: Any) -> dict:
    headers = {}
    if not text:
        return headers
    for line in str(text).splitlines():
        line = line.strip()
        if not line or ':' not in line:
            continue
        key, value = line.split(':', 1)
        headers[key.strip()] = value.strip()
    return headers


def compare_value(actual, operator: str, expected=None, var_type: str = 'string') -> bool:
    op = (operator or '').replace(' ', '_').replace('-', '_').lower()
    if op in ('empty', 'is_empty'):
        return actual in (None, '', [], {})
    if op in ('not_empty', 'is_not_empty'):
        return actual not in (None, '', [], {})
    if actual is None:
        actual = ''
    if expected is None:
        expected = ''
    if op in ('is', 'equal', 'equals', '='):
        return str(actual) == str(expected)
    if op in ('is_not', 'not_equal', 'not_equals', '!='):
        return str(actual) != str(expected)
    if op in ('contains',):
        return str(expected) in str(actual)
    if op in ('not_contains', 'not contains'):
        return str(expected) not in str(actual)
    if op in ('start_with', 'starts_with', 'prefix'):
        return str(actual).startswith(str(expected))
    if op in ('end_with', 'ends_with', 'suffix'):
        return str(actual).endswith(str(expected))
    if op in ('in',):
        return str(actual) in str(expected)
    if op in ('not_in',):
        return str(actual) not in str(expected)
    try:
        left, right = float(actual), float(expected)
    except (TypeError, ValueError):
        left, right = None, None
    if left is not None:
        if op in ('>', 'gt', 'larger_than'):
            return left > right
        if op in ('<', 'lt', 'less_than'):
            return left < right
        if op in ('>=', 'ge', 'larger_than_or_equal'):
            return left >= right
        if op in ('<=', 'le', 'less_than_or_equal'):
            return left <= right
    return False


def match_case(obj: dict, case: dict) -> bool:
    conds = case.get('conditions') or []
    if not conds:
        return False
    flags = []
    for cond in conds:
        selector = cond.get('variable_selector') or []
        actual = sel(obj, selector[0], *selector[1:]) if selector else ''
        flags.append(compare_value(
            actual,
            cond.get('comparison_operator') or cond.get('comparisonOperator') or '',
            cond.get('value'),
            cond.get('varType') or cond.get('var_type') or 'string',
        ))
    logic = (case.get('logical_operator') or case.get('logicalOperator') or 'and').lower()
    return any(flags) if logic == 'or' else all(flags)


def pick_switch_handle(obj: dict, cases: Sequence[dict], else_handle: str = 'false') -> str:
    for case in cases:
        if match_case(obj, case):
            return str(case.get('case_id') or case.get('id') or 'true')
    return else_handle
