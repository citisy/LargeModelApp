#!/usr/bin/env python3
"""Convert a Dify workflow YAML into a runnable `workflows.skeletons` Python module.

Usage:
    python dify2module.py [xxx].yml
    python dify2module.py [dir] -o [out_dir]

Generated code follows `components.base.BaseModelWithoutDb` plus pipeline style:
each Dify node becomes a Module. The DAG is layered topologically
- serial steps - Pipeline
- fan-out - MultiThreadPipeline
- branches - SwitchPipeline
- iteration - Sequential
"""
import argparse
import ast
import json
import re
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from utils import os_lib


class DifyGraph:
    def __init__(self, dsl: dict):
        self.dsl = dsl
        self.app = dsl.get('app') or {}
        self.workflow = dsl.get('workflow') or {}
        graph = self.workflow.get('graph') or {}
        self.raw_nodes = graph.get('nodes') or []
        self.raw_edges = graph.get('edges') or []
        self.app_name = self.app.get('name') or 'DifyWorkflow'

        self.nodes: Dict[str, dict] = {}
        for node in self.raw_nodes:
            self.nodes[str(node['id'])] = node

        self.edges: List[dict] = []
        for edge in self.raw_edges:
            self.edges.append({
                'source': str(edge.get('source')),
                'target': str(edge.get('target')),
                'sourceHandle': str(edge.get('sourceHandle') or 'source'),
                'sourceType': ((edge.get('data') or {}).get('sourceType')
                               or self.type_of(str(edge.get('source')))),
                'targetType': ((edge.get('data') or {}).get('targetType')
                               or self.type_of(str(edge.get('target')))),
            })

        self.children: Dict[str, List[str]] = defaultdict(list)
        for nid, node in self.nodes.items():
            parent = node.get('parentId') or node.get('parent_id')
            if parent:
                self.children[str(parent)].append(nid)

        self.out: Dict[str, List[dict]] = defaultdict(list)
        self.inn: Dict[str, List[dict]] = defaultdict(list)
        for edge in self.edges:
            self.out[edge['source']].append(edge)
            self.inn[edge['target']].append(edge)

    def type_of(self, nid: str) -> str:
        node = self.nodes.get(str(nid)) or {}
        return ((node.get('data') or {}).get('type') or '')

    def data_of(self, nid: str) -> dict:
        node = self.nodes.get(str(nid)) or {}
        return node.get('data') or {}

    def title_of(self, nid: str) -> str:
        return self.data_of(nid).get('title') or self.type_of(nid) or str(nid)

    def is_child(self, nid: str) -> bool:
        node = self.nodes.get(str(nid)) or {}
        return bool(node.get('parentId') or node.get('parent_id'))

    def top_level_ids(self) -> List[str]:
        return [nid for nid in self.nodes if not self.is_child(nid)]

    def sort_key(self, nid: str):
        node = self.nodes.get(nid) or {}
        pos = node.get('position') or {}
        return (float(pos.get('x') or 0), float(pos.get('y') or 0), nid)

    def start_id(self, ids: Optional[Iterable[str]] = None) -> Optional[str]:
        pool = list(ids) if ids is not None else self.top_level_ids()
        for nid in pool:
            if self.type_of(nid) == 'start':
                return nid
        return pool[0] if pool else None

    def end_ids(self, ids: Optional[Iterable[str]] = None) -> List[str]:
        pool = list(ids) if ids is not None else self.top_level_ids()
        return [nid for nid in pool if self.type_of(nid) in ('end', 'answer')]


class CodeGenerator:
    type_prefix = {
        'start': 'Start',
        'end': 'End',
        'answer': 'Answer',
        'llm': 'Llm',
        'code': 'Code',
        'template-transform': 'Tpl',
        'http-request': 'Http',
        'tool': 'Tool',
        'if-else': 'IfElse',
        'iteration': 'Iter',
        'loop': 'Loop',
        'question-classifier': 'Classifier',
        'variable-aggregator': 'Agg',
        'assigner': 'Assign',
        'parameter-extractor': 'ParamExt',
        'knowledge-retrieval': 'Kb',
        'document-extractor': 'DocExt',
        'list-operator': 'ListOp',
        'agent': 'Agent',
        'iteration-start': 'IterStart',
        'loop-start': 'LoopStart',
    }
    thinking_hint = re.compile(r'(r1|reasoner|thinking|deepseek[-_]?r1)', re.I)

    def __init__(self, files: Sequence[str | Path] | str | Path):
        if isinstance(files, (str, Path)):
            files = [files]
        self.files = [Path(p) for p in files]
        configs = [os_lib.loader.load_yaml(p) for p in self.files]
        self.module_index = self.collect_module_index(configs)
        self.graphs = [DifyGraph(cfg) for cfg in configs]
        self.g: Optional[DifyGraph] = None
        self.class_names: Dict[str, str] = {}
        self.warnings: List[str] = []

    def _use(self, graph: DifyGraph):
        self.g = graph
        self.class_names = {
            nid: self.cls_name(self.g.type_of(nid), nid) for nid in self.g.nodes
        }
        self.warnings = []

    @classmethod
    def cls_name(cls, node_type: str, node_id: Any) -> str:
        prefix = cls.type_prefix.get(node_type, 'Node')
        safe_id = re.sub(r'\W+', '_', str(node_id)).strip('_')
        return f'{prefix}_{safe_id}'

    @staticmethod
    def to_py_string(text: Any) -> str:
        text = '' if text is None else str(text)
        if '"""' not in text and '\\' not in text:
            return '"""' + text + '"""'
        return repr(text)

    @staticmethod
    def safe_filename(name: str) -> str:
        name = (name or 'workflow').strip()
        name = re.sub(r'[<>:"/\\|?*]', '_', name)
        name = re.sub(r'\s+', '_', name)
        name = name.replace('-', '_')
        return name or 'workflow'

    @staticmethod
    def env_model_key(model_name: str) -> str:
        return 'DIFY_MODEL_' + re.sub(r'\W+', '_', model_name).strip('_')

    @staticmethod
    def indent_block(text: str, n: int) -> str:
        pad = ' ' * n
        return '\n'.join(pad + line if line else line for line in text.splitlines())

    @staticmethod
    def join_blocks(*blocks: str) -> str:
        """Join top-level class / constant blocks with two blank lines."""
        cleaned = []
        for block in blocks:
            if not block:
                continue
            text = str(block).strip('\n')
            if text.strip():
                cleaned.append(text)
        return '\n\n\n'.join(cleaned) + '\n'

    @classmethod
    def collect_module_index(cls, configs: List[dict]) -> Dict[str, str]:
        index = {}
        for dsl in configs:
            name = ((dsl.get('app') or {}).get('name')) or 'DifyWorkflow'
            mod = cls.safe_filename(name)
            index[name] = mod
            index[mod] = mod
        return index

    # ----- DAG compile -----

    def compile(self, ids: Sequence[str]) -> dict:
        id_set = set(ids)
        indeg = {nid: 0 for nid in id_set}
        adj = defaultdict(list)
        for edge in self.g.edges:
            s, t = edge['source'], edge['target']
            if s in id_set and t in id_set:
                adj[s].append(t)
                indeg[t] += 1

        remaining = set(id_set)
        steps = []
        while remaining:
            ready = sorted(
                [nid for nid in remaining if indeg.get(nid, 0) <= 0],
                key=self.g.sort_key,
            )
            if not ready:
                ready = [min(remaining, key=self.g.sort_key)]
                self.warnings.append(f'Cycle or inbound-edge anomaly, force-expanding node {ready[0]}')

            splits = [nid for nid in ready if self.g.type_of(nid) in ('if-else', 'question-classifier')]
            normals = [nid for nid in ready if nid not in splits]

            if splits and not normals:
                for nid in splits[:1]:
                    step, consumed = self._compile_split(nid, remaining)
                    steps.append(step)
                    self._consume(remaining, indeg, adj, consumed)
                continue

            group = normals or splits
            if len(group) == 1 and self.g.type_of(group[0]) == 'iteration':
                steps.append(self._compile_iteration(group[0]))
                self._consume(remaining, indeg, adj, {group[0]} | set(self.g.children.get(group[0], [])))
                continue
            if len(group) == 1 and self.g.type_of(group[0]) == 'loop':
                steps.append(self._compile_loop(group[0]))
                self._consume(remaining, indeg, adj, {group[0]} | set(self.g.children.get(group[0], [])))
                continue
            if len(group) == 1:
                steps.append({'kind': 'node', 'id': group[0]})
                self._consume(remaining, indeg, adj, {group[0]})
            else:
                steps.append({
                    'kind': 'parallel',
                    'children': [{'kind': 'node', 'id': nid} for nid in group],
                })
                self._consume(remaining, indeg, adj, set(group))
        return {'kind': 'seq', 'children': steps}

    def _consume(self, remaining, indeg, adj, consumed):
        consumed = set(consumed) & remaining
        remaining -= consumed
        for nid in consumed:
            for nxt in adj.get(nid, []):
                if nxt in remaining:
                    indeg[nxt] = max(0, indeg.get(nxt, 0) - 1)

    def _compile_split(self, nid: str, remaining: set) -> Tuple[dict, set]:
        handles = defaultdict(list)
        for edge in self.g.out.get(nid, []):
            if edge['target'] in remaining:
                handles[edge['sourceHandle']].append(edge['target'])

        reach = {}
        for handle, targets in handles.items():
            seen = set()
            dq = deque(targets)
            while dq:
                cur = dq.popleft()
                if cur not in remaining or cur in seen:
                    continue
                seen.add(cur)
                for edge in self.g.out.get(cur, []):
                    if edge['target'] in remaining:
                        dq.append(edge['target'])
            reach[handle] = seen

        count = defaultdict(int)
        for nodes in reach.values():
            for n in nodes:
                count[n] += 1
        joins = {n for n, c in count.items() if c >= 2}

        exclusive = {}
        for handle, targets in handles.items():
            got = set()
            dq = deque(targets)
            seen = set()
            while dq:
                cur = dq.popleft()
                if cur in seen or cur not in remaining or cur in joins:
                    continue
                seen.add(cur)
                got.add(cur)
                for edge in self.g.out.get(cur, []):
                    nxt = edge['target']
                    if nxt not in joins:
                        dq.append(nxt)
            exclusive[handle] = got

        cases = []
        for handle, targets in handles.items():
            sub_ids = exclusive.get(handle, set())
            inner = self.compile(sorted(sub_ids, key=self.g.sort_key)) if sub_ids else {'kind': 'seq', 'children': []}
            cases.append({'handle': handle, 'inner': inner, 'targets': targets})

        consumed = {nid}
        for nodes in exclusive.values():
            consumed |= nodes
        return {'kind': 'switch', 'id': nid, 'cases': cases}, consumed

    def _compile_iteration(self, nid: str) -> dict:
        child_ids = [cid for cid in self.g.children.get(nid, []) if self.g.type_of(cid) != 'iteration-start']
        inner = self.compile(child_ids) if child_ids else {'kind': 'seq', 'children': []}
        return {'kind': 'iteration', 'id': nid, 'inner': inner}

    def _compile_loop(self, nid: str) -> dict:
        child_ids = [cid for cid in self.g.children.get(nid, []) if self.g.type_of(cid) != 'loop-start']
        inner = self.compile(child_ids) if child_ids else {'kind': 'seq', 'children': []}
        return {'kind': 'loop', 'id': nid, 'inner': inner}

    # ----- emit constructors -----

    def emit_ctor(self, step: dict, indent: int = 12) -> str:
        pad = ' ' * indent
        kind = step['kind']
        if kind == 'node':
            return f"{pad}{self.class_names[step['id']]}(),"
        if kind == 'seq':
            children = step.get('children') or []
            if not children:
                name = step.get('name')
                extra = f', name={name!r}' if name else ''
                return f"{pad}skeletons.Module(name={name!r})," if name else f"{pad}skeletons.Module(),"
            if len(children) == 1 and not step.get('name'):
                return self.emit_ctor(children[0], indent)
            lines = [f"{pad}skeletons.Pipeline("]
            for ch in children:
                lines.append(self.emit_ctor(ch, indent + 4))
            if step.get('name'):
                lines.append(f"{pad}    name={step['name']!r},")
            lines.append(f"{pad}),")
            return '\n'.join(lines)
        if kind == 'parallel':
            lines = [f"{pad}skeletons.MultiThreadPipeline("]
            for ch in step.get('children') or []:
                lines.append(self.emit_ctor(ch, indent + 4))
            lines.append(f"{pad}),")
            return '\n'.join(lines)
        if kind in ('switch', 'iteration', 'loop'):
            return f"{pad}{self.class_names[step['id']]}(),"
        raise ValueError(kind)

    # ----- class generation -----

    def generate(self, gen_test_code=False) -> List[str]:
        codes = []
        for graph in self.graphs:
            self._use(graph)
            top = self.g.top_level_ids()
            tree = self.compile(top)
            classes = self._gen_all_node_classes(tree)
            parts = [
                self._header(gen_test_code=gen_test_code),
                classes,
                self._gen_model_class(tree),
            ]
            if gen_test_code:
                parts.append(self._gen_main())
            codes.append(self.join_blocks(*parts))
        return codes

    def to_files(self, out_dir: str = None, out_files: List[str] = None, gen_test_code=False) -> List[Path]:
        codes = self.generate(gen_test_code=gen_test_code)
        if out_files is not None and len(out_files) != len(codes):
            raise ValueError(f'out_files length {len(out_files)} != generated modules {len(codes)}')

        if out_files:
            paths = [Path(p) for p in out_files]
            if out_dir:
                root = Path(out_dir)
                paths = [p if p.is_absolute() else root / p for p in paths]
        elif out_dir is not None:
            root = Path(out_dir)
            paths = [root / f'{self.safe_filename(name)}.py' for name in self.app_names]
        else:
            raise

        written = []
        for path, code in zip(paths, codes):
            ast.parse(code)
            os_lib.mk_parent_dir(path)
            path.write_text(code, encoding='utf-8')
            written.append(path)

        if out_files is None or out_dir:
            root = Path(out_dir) if out_dir else written[0].parent
            self._write_package_init(root)
        return written

    def _write_package_init(self, dst_dir: Path):
        init_map = {name: self.safe_filename(name) for name in self.app_names}
        lines = ['# Auto-generated Dify workflow modules\n', 'WORKFLOWS = {\n']
        for name, mod in init_map.items():
            lines.append(f'    {name!r}: {mod!r},\n')
        lines.append('}\n\n')
        lines.append('def get_model(name):\n')
        lines.append('    import importlib\n')
        lines.append('    mod_name = WORKFLOWS[name]\n')
        lines.append("    mod = importlib.import_module(f'.{mod_name}', __package__)\n")
        lines.append('    return mod.Model\n')
        (dst_dir / '__init__.py').write_text(''.join(lines), encoding='utf-8')

    @property
    def app_names(self) -> List[str]:
        return [g.app_name for g in self.graphs]

    def _used_types(self) -> set:
        return {self.g.type_of(nid) for nid in self.g.nodes}

    def _header(self, gen_test_code=False) -> str:
        types = self._used_types()
        helpers = ['sel', 'set_node']
        if types & {'llm', 'http-request', 'tool'}:
            helpers.append('render_prompt')
        if 'template-transform' in types:
            helpers.append('render_template')
        if 'http-request' in types:
            helpers.append('parse_http_headers')
        if 'if-else' in types:
            helpers.extend(['match_case', 'pick_switch_handle'])
        helpers.append('dify_register_modules')
        helper_lines = ',\n    '.join(helpers)
        volc = 'from components.sdks.openai import Volcengine\n' if 'llm' in types else ''
        json_imp = 'import json\n' if gen_test_code or types & {'http-request', 'tool'} else ''
        os_imp = 'import os\n' if 'llm' in types else ''
        warns = ''
        if self.warnings:
            warns = '\n'.join(f'# warning: {w}' for w in self.warnings) + '\n'
        return f'''# -*- coding: utf-8 -*-
"""{self.g.app_name}

Auto-converted from a Dify workflow by dify2module.py.
Override the actual LLM model id with env vars, e.g. DIFY_MODEL_doubao_1_6.
"""
{json_imp}{os_imp}

from components.base import BaseModelWithoutDb
from components.dify_helper import (
    {helper_lines},
)
{volc}from workflows import skeletons

{warns}'''

    def _walk_steps(self, step: dict) -> Iterable[dict]:
        yield step
        for key in ('children', 'cases'):
            for ch in step.get(key) or []:
                if isinstance(ch, dict) and 'kind' in ch:
                    yield from self._walk_steps(ch)
                elif isinstance(ch, dict) and 'inner' in ch:
                    yield from self._walk_steps(ch['inner'])
        if 'inner' in step and isinstance(step['inner'], dict):
            yield from self._walk_steps(step['inner'])

    def _needed_ids(self, tree: dict) -> List[str]:
        needed = []
        seen = set()
        for step in self._walk_steps(tree):
            nid = step.get('id')
            if nid and nid not in seen:
                seen.add(nid)
                needed.append(nid)
            if step.get('kind') == 'iteration':
                for cid in self.g.children.get(step['id'], []):
                    if cid not in seen:
                        seen.add(cid)
                        needed.append(cid)
            if step.get('kind') == 'loop':
                for cid in self.g.children.get(step['id'], []):
                    if cid not in seen:
                        seen.add(cid)
                        needed.append(cid)
        # Always emit top-level nodes, even if compile skipped them.
        for nid in self.g.top_level_ids():
            if nid not in seen:
                needed.append(nid)
        return needed

    def _gen_all_node_classes(self, tree: dict) -> str:
        parts = []
        emitted = set()
        for nid in self._needed_ids(tree):
            if nid in emitted:
                continue
            emitted.add(nid)
            ntype = self.g.type_of(nid)
            fn = {
                'start': self._gen_start,
                'end': self._gen_end,
                'answer': self._gen_end,
                'llm': self._gen_llm,
                'code': self._gen_code,
                'template-transform': self._gen_template,
                'http-request': self._gen_http,
                'tool': self._gen_tool,
                'if-else': self._gen_ifelse,
                'question-classifier': self._gen_classifier,
                'iteration': self._gen_iteration_cls,
                'loop': self._gen_loop_cls,
                'variable-aggregator': self._gen_aggregator,
                'assigner': self._gen_assigner,
                'iteration-start': self._gen_passthrough,
                'loop-start': self._gen_passthrough,
            }.get(ntype, self._gen_stub)
            # Attach compiled tree info for switch / iteration nodes.
            extra = None
            for step in self._walk_steps(tree):
                if step.get('id') == nid:
                    extra = step
                    break
            parts.append(fn(nid, extra))
        return self.join_blocks(*parts)

    def _register_deco(self) -> str:
        table = json.dumps(self.g.app_name, ensure_ascii=False)
        return f'@dify_register_modules.add_register(table_name={table})'

    def _cls_head(self, nid: str, base: str, extra_attrs: str = '') -> str:
        title = self.g.title_of(nid)
        desc = (self.g.data_of(nid).get('desc') or '').strip()
        doc = title if not desc else f'{title}\n    {desc}'
        attrs = extra_attrs if extra_attrs.endswith('\n') or extra_attrs == '' else extra_attrs + '\n'
        return (
            f'{self._register_deco()}\n'
            f'class {self.class_names[nid]}({base}):\n'
            f'    """{doc}"""\n'
            f'    node_id = {str(nid)!r}\n'
            f'{attrs}\n'
        )

    def _gen_start(self, nid: str, _step=None) -> str:
        variables = self.g.data_of(nid).get('variables') or []
        body = ['    def on_process(self, obj, **kwargs):']
        if not variables:
            body += ['        set_node(obj, self.node_id)', '        return obj']
            return self._cls_head(nid, 'skeletons.Module') + '\n'.join(body) + '\n'
        body += ['        set_node(', '            obj,', '            self.node_id,']
        for var in variables:
            key = var.get('variable') or var.get('name')
            if not key:
                continue
            default = var.get('default')
            if default is None:
                default = 0 if var.get('type') == 'number' else ''
            body.append(f'            {key}=obj.get({key!r}, {default!r}),')
        body += ['        )', '        return obj']
        return self._cls_head(nid, 'skeletons.Module') + '\n'.join(body) + '\n'

    def _gen_end(self, nid: str, _step=None) -> str:
        outputs = self.g.data_of(nid).get('outputs') or []
        body = ['    def on_process(self, obj, **kwargs):']
        if not outputs:
            body.append('        return obj')
            return self._cls_head(nid, 'skeletons.Module') + '\n'.join(body) + '\n'
        for item in outputs:
            selector = item.get('value_selector') or item.get('valueSelector') or []
            var = item.get('variable') or item.get('name') or (selector[-1] if selector else 'output')
            if selector:
                node_id, keys = selector[0], selector[1:]
                key_args = ', '.join(repr(k) for k in keys)
                extra = f', {key_args}' if key_args else ''
                body.append(f'        obj[{var!r}] = sel(obj, {str(node_id)!r}{extra})')
            else:
                body.append(f'        obj[{var!r}] = obj.get({var!r}, \'\')')
        body.append('        return obj')
        return self._cls_head(nid, 'skeletons.Module') + '\n'.join(body) + '\n'

    def _gen_llm(self, nid: str, _step=None) -> str:
        data = self.g.data_of(nid)
        model = ((data.get('model') or {}).get('name')) or 'gpt-4'
        params = ((data.get('model') or {}).get('completion_params')) or {}
        prompts = data.get('prompt_template') or []
        disable_thinking = not bool(self.thinking_hint.search(str(model) + str(data.get('title') or '')))
        env_key = self.env_model_key(model)
        param_repr = ', '.join(f'{k}={v!r}' for k, v in params.items() if k != 'stop')
        request_extra = f', {param_repr}' if param_repr else ''

        consts = []
        msg_parts = []
        for i, item in enumerate(prompts):
            role = item.get('role') or 'user'
            text = item.get('text') or ''
            if not str(text).strip() and role == 'assistant':
                continue
            cname = f'{self.class_names[nid]}_{role}_{i}'.upper()
            consts.append(f'{cname} = {self.to_py_string(text)}')
            msg_parts.append((role, cname))

        sys_name = next((c for r, c in msg_parts if r == 'system'), None)
        user_name = next((c for r, c in msg_parts if r == 'user'), None)
        multi = len(msg_parts) > 2 or any(r == 'assistant' for r, _ in msg_parts)

        head = self._cls_head(
            nid, 'Volcengine',
            extra_attrs=(
                f'    model = os.getenv({env_key!r}, {model!r})\n'
                f'    disable_thinking = {disable_thinking}\n'
            ),
        )
        body = ['    def on_process(self, obj, **kwargs):']
        if multi:
            body.append('        messages = [')
            for role, cname in msg_parts:
                body.append(f'            {{"role": {role!r}, "content": render_prompt({cname}, obj)}},')
            body.append('        ]')
            body.append(f'        text = self.request(messages=messages{request_extra})')
        elif sys_name and user_name:
            body.append(f'        sys = render_prompt({sys_name}, obj)')
            body.append(f'        user = render_prompt({user_name}, obj)')
            body.append(f'        text = self.request(sys=sys, user=user{request_extra})')
        elif user_name:
            body.append(f'        user = render_prompt({user_name}, obj)')
            body.append(f'        text = self.request(messages=[{{"role": "user", "content": user}}]{request_extra})')
        elif sys_name:
            body.append(f'        sys = render_prompt({sys_name}, obj)')
            body.append(f'        text = self.request(sys=sys, user=\'\'{request_extra})')
        else:
            body.append("        text = self.request(sys='', user='')")
        vision = data.get('vision') or {}
        if vision.get('enabled'):
            body.append('        # TODO: vision.enabled=True; add images to messages if needed')
        body.append('        set_node(obj, self.node_id, text=text)')
        body.append('        return obj')
        class_src = head + '\n'.join(body)
        if consts:
            return self.join_blocks('\n'.join(consts), class_src)
        return class_src + '\n'

    def _gen_code(self, nid: str, _step=None) -> str:
        data = self.g.data_of(nid)
        code = data.get('code') or 'def main(**kwargs):\n    return {}'
        variables = data.get('variables') or []
        head = self._cls_head(
            nid, 'skeletons.Module',
            extra_attrs=f'    CODE = {self.to_py_string(code)}\n',
        )
        lines = [
            '    def on_process(self, obj, **kwargs):',
            '        inputs = {}',
        ]
        for var in variables:
            name = var.get('variable') or var.get('name')
            selector = var.get('value_selector') or var.get('valueSelector') or []
            if not name:
                continue
            if selector:
                extra = ', '.join(repr(str(x)) for x in selector[1:])
                extra = f', {extra}' if extra else ''
                lines.append(f'        inputs[{name!r}] = sel(obj, {str(selector[0])!r}{extra})')
            else:
                lines.append(f'        inputs[{name!r}] = obj.get({name!r}, \'\')')
        lines += [
            '        ns = {}',
            '        exec(self.CODE, ns, ns)',
            "        fn = ns.get('main')",
            '        result = fn(**inputs) if callable(fn) else {}',
            '        if not isinstance(result, dict):',
            "            result = {'result': result}",
            '        set_node(obj, self.node_id, **result)',
            '        return obj',
        ]
        return head + '\n'.join(lines) + '\n'

    def _gen_template(self, nid: str, _step=None) -> str:
        data = self.g.data_of(nid)
        template = data.get('template') or ''
        variables = data.get('variables') or []
        head = self._cls_head(
            nid, 'skeletons.Module',
            extra_attrs=f'    TEMPLATE = {self.to_py_string(template)}\n',
        )
        lines = [
            '    def on_process(self, obj, **kwargs):',
            '        variables = {}',
        ]
        for var in variables:
            name = var.get('variable') or var.get('name')
            selector = var.get('value_selector') or var.get('valueSelector') or []
            if not name:
                continue
            if selector:
                extra = ', '.join(repr(str(x)) for x in selector[1:])
                extra = f', {extra}' if extra else ''
                lines.append(f'        variables[{name!r}] = sel(obj, {str(selector[0])!r}{extra})')
            else:
                lines.append(f'        variables[{name!r}] = obj.get({name!r}, \'\')')
        lines += [
            '        output = render_template(self.TEMPLATE, obj, variables)',
            '        set_node(obj, self.node_id, output=output)',
            '        return obj',
        ]
        return head + '\n'.join(lines) + '\n'

    def _gen_http(self, nid: str, _step=None) -> str:
        data = self.g.data_of(nid)
        retry = data.get('retry_config') or {}
        retry_count = int(retry.get('max_retries') or 3)
        retry_wait = float(retry.get('retry_interval') or 100) / 1000.0
        url = data.get('url') or ''
        method = (data.get('method') or 'GET').upper()
        headers = data.get('headers') or ''
        params = data.get('params') or ''
        body = data.get('body') or {}
        body_type = body.get('type') or 'none'
        body_data = body.get('data')
        if isinstance(body_data, list):
            # Key-value list for json / form; prefer the value field.
            values = [str(item.get('value') or '') for item in body_data if isinstance(item, dict)]
            body_tpl = values[0] if len(values) == 1 else json.dumps(
                {item.get('key'): item.get('value') for item in body_data if isinstance(item, dict)},
                ensure_ascii=False,
            )
        elif isinstance(body_data, dict):
            body_tpl = json.dumps(body_data, ensure_ascii=False)
        else:
            body_tpl = '' if body_data is None else str(body_data)
        timeout = data.get('timeout') or {}
        timeout_s = timeout.get('max_read_timeout') or timeout.get('max_connect_timeout') or 0
        timeout_s = float(timeout_s) if timeout_s else None

        head = self._cls_head(
            nid, 'skeletons.RetryModule',
            extra_attrs=(
                f'    retry_count = {retry_count}\n'
                f'    retry_wait = {retry_wait}\n'
                f'    method = {method!r}\n'
                f'    url_tpl = {self.to_py_string(url)}\n'
                f'    headers_tpl = {self.to_py_string(headers)}\n'
                f'    params_tpl = {self.to_py_string(params)}\n'
                f'    body_tpl = {self.to_py_string(body_tpl)}\n'
                f'    body_type = {body_type!r}\n'
                f'    timeout = {timeout_s!r}\n'
            ),
        )
        return head + '''    def on_process(self, obj, **kwargs):
        import requests
        url = render_prompt(self.url_tpl, obj)
        headers = parse_http_headers(render_prompt(self.headers_tpl, obj))
        params = render_prompt(self.params_tpl, obj)
        body = render_prompt(self.body_tpl, obj)
        req_kwargs = dict(headers=headers, timeout=self.timeout or 60)
        if params:
            req_kwargs['params'] = params
        if self.body_type == 'json' and body:
            try:
                req_kwargs['json'] = json.loads(body)
            except json.JSONDecodeError:
                req_kwargs['data'] = body.encode('utf-8')
        elif body:
            req_kwargs['data'] = body.encode('utf-8')
        resp = requests.request(self.method, url, **req_kwargs)
        set_node(
            obj, self.node_id,
            body=resp.text,
            status_code=resp.status_code,
            headers=dict(resp.headers),
        )
        return obj
'''

    def _gen_tool(self, nid: str, _step=None) -> str:
        data = self.g.data_of(nid)
        provider = data.get('provider_name') or data.get('provider_id') or ''
        tool_name = data.get('tool_name') or data.get('tool_label') or ''
        provider_type = data.get('provider_type') or ''
        params = data.get('tool_parameters') or {}
        module_file = self.module_index.get(provider) or self.module_index.get(tool_name)
        head = self._cls_head(
            nid, 'skeletons.Module',
            extra_attrs=(
                f'    provider_name = {provider!r}\n'
                f'    tool_name = {tool_name!r}\n'
                f'    provider_type = {provider_type!r}\n'
            ),
        )
        lines = ['    def on_process(self, obj, **kwargs):', '        params = {}']
        for key, spec in params.items():
            if not isinstance(spec, dict):
                lines.append(f'        params[{key!r}] = {spec!r}')
                continue
            typ = spec.get('type') or 'mixed'
            value = spec.get('value')
            if typ == 'constant':
                lines.append(f'        params[{key!r}] = {value!r}')
            else:
                lines.append(f'        params[{key!r}] = render_prompt({self.to_py_string(value)}, obj)')
        if provider_type == 'workflow' or module_file:
            if not module_file:
                self.warnings.append(
                    f'Tool node {self.g.title_of(nid)} ({provider}/{tool_name}) '
                    f'is a workflow tool; import that module so it registers as {provider!r}'
                )
            lines += [
                "        inner_module = dify_register_modules.get('Model', self.provider_name)()",
                "        task_id = kwargs.get('task_id') or 'tool'",
                "        ret = inner_module({'task_id': task_id, 'kwargs': params}, task_id=task_id)",
                "        data = ret.get('data') if isinstance(ret, dict) else ret",
                '        data = data or {}',
                "        text = data.get('body') or data.get('result') or data.get('text')",
                '        if text is None:',
                '            text = json.dumps(data, ensure_ascii=False)',
                '        payload = dict(data) if isinstance(data, dict) else {}',
                "        payload.update(text=text, json=data)",
                '        set_node(obj, self.node_id, **payload)',
                '        return obj',
            ]
        else:
            self.warnings.append(f'Tool node {self.g.title_of(nid)} ({provider}/{tool_name}) has no converted workflow module; emitting a stub')
            lines += [
                '        # Non-workflow tool (plugin / API); implement the call yourself.',
                "        raise NotImplementedError(f'Unconverted Dify Tool: {self.provider_name}/{self.tool_name}, params={params}')",
            ]
        return head + '\n'.join(lines) + '\n'

    def _gen_ifelse(self, nid: str, step=None) -> str:
        data = self.g.data_of(nid)
        cases = data.get('cases') or []
        step = step or {'cases': []}
        handle_ctors = []
        for case in (step.get('cases') or []):
            handle = case['handle']
            inner = dict(case['inner'])
            inner['name'] = handle
            ctor = self.emit_ctor(inner, indent=12)
            handle_ctors.append(ctor)
        if not handle_ctors:
            # No compiled edges; emit empty branches from cases + false.
            for case in cases:
                hid = str(case.get('case_id') or 'true')
                handle_ctors.append(f"            skeletons.Module(name={hid!r}),")
            handle_ctors.append("            skeletons.Module(name='false'),")
        head = self._cls_head(
            nid, 'skeletons.SwitchPipeline',
            extra_attrs=f'    CASES = {repr(cases)}\n',
        )
        init = (
                '    def __init__(self, **kwargs):\n'
                '        super().__init__(\n'
                + '\n'.join(handle_ctors) + '\n'
                                            '            **kwargs\n'
                                            '        )\n'
                                            '    def switch(self, obj, **kwargs):\n'
                                            "        return pick_switch_handle(obj, self.CASES, else_handle='false')\n"
        )
        return head + init

    def _gen_classifier(self, nid: str, step=None) -> str:
        data = self.g.data_of(nid)
        classes = data.get('classes') or []
        self.warnings.append(f'Question classifier {self.g.title_of(nid)} needs an LLM classify step; currently uses first class / default')
        step = step or {'cases': []}
        handle_ctors = []
        for case in (step.get('cases') or []):
            inner = dict(case['inner'])
            inner['name'] = case['handle']
            handle_ctors.append(self.emit_ctor(inner, indent=12))
        if not handle_ctors:
            for cls in classes:
                handle_ctors.append(f"            skeletons.Module(name={str(cls.get('id') or cls.get('name'))!r}),")
        head = self._cls_head(
            nid, 'skeletons.SwitchPipeline',
            extra_attrs=f'    CLASSES = {repr(classes)}\n',
        )
        return head + (
                '    def __init__(self, **kwargs):\n'
                '        super().__init__(\n'
                + '\n'.join(handle_ctors) + '\n'
                                            '            **kwargs\n'
                                            '        )\n'
                                            '    def switch(self, obj, **kwargs):\n'
                                            "        # TODO: run an LLM to emit a class id; currently returns the first class or 'false'\n"
                                            "        if self.CLASSES:\n"
                                            "            return str(self.CLASSES[0].get('id') or self.CLASSES[0].get('name') or 'false')\n"
                                            "        return 'false'\n"
        )

    def _gen_iteration_cls(self, nid: str, step=None) -> str:
        data = self.g.data_of(nid)
        iterator = data.get('iterator_selector') or data.get('iteratorSelector') or []
        output_sel = data.get('output_selector') or data.get('outputSelector') or []
        start_id = data.get('start_node_id') or data.get('startNodeId')
        if not start_id:
            for cid in self.g.children.get(nid, []):
                if self.g.type_of(cid) == 'iteration-start':
                    start_id = cid
                    break
        step = step or {'inner': {'kind': 'seq', 'children': []}}
        inner_ctor = self.emit_ctor(step['inner'], indent=12)
        it_node = repr(str(iterator[0])) if iterator else "''"
        it_keys = ', '.join(repr(str(x)) for x in iterator[1:]) if len(iterator) > 1 else ''
        it_keys_arg = f', {it_keys}' if it_keys else ''
        out_node = repr(str(output_sel[0])) if output_sel else "''"
        out_keys = ', '.join(repr(str(x)) for x in output_sel[1:]) if len(output_sel) > 1 else ''
        out_keys_arg = f', {out_keys}' if out_keys else ''
        in_name = self.class_names[nid] + 'Input'
        out_name = self.class_names[nid] + 'Output'
        deco = self._register_deco()
        input_cls = f'''{deco}
class {in_name}(skeletons.BaseSequentialInput):
    """{self.g.title_of(nid)} iteration input"""

    def on_process(self, obj, **kwargs):
        import copy
        items = sel(obj, {it_node}{it_keys_arg}, default=[])
        if items in ('', None):
            items = []
        if not isinstance(items, (list, tuple)):
            items = [items]
        for i, item in enumerate(items):
            item_obj = copy.deepcopy(obj)
            set_node(item_obj, {str(start_id)!r}, item=item, index=i)
            yield item_obj
'''
        output_cls = f'''{deco}
class {out_name}(skeletons.BaseSequentialOutput):
    """{self.g.title_of(nid)} iteration output"""

    def on_process(self, objs, raw_obj=None, **kwargs):
        outputs = [sel(o, {out_node}{out_keys_arg}) for o in (objs or [])]
        set_node(raw_obj if raw_obj is not None else {{}}, {str(nid)!r}, output=outputs)
        return raw_obj
'''
        iter_cls = f'''{deco}
class {self.class_names[nid]}(skeletons.Sequential):
    """{self.g.title_of(nid)}"""
    node_id = {str(nid)!r}

    def __init__(self, **kwargs):
        super().__init__(
            {in_name}(),
{inner_ctor}
            {out_name}(),
            **kwargs
        )
'''
        return self.join_blocks(input_cls, output_cls, iter_cls)

    def _gen_loop_cls(self, nid: str, step=None) -> str:
        data = self.g.data_of(nid)
        step = step or {'inner': {'kind': 'seq', 'children': []}}
        inner_ctor = self.emit_ctor(step['inner'], indent=12)
        self.warnings.append(f'Loop node {self.g.title_of(nid)}: verify check() against the Dify loop condition')
        head = self._cls_head(nid, 'skeletons.LoopPipeline')
        return head + (
                '    def __init__(self, **kwargs):\n'
                '        super().__init__(\n'
                + inner_ctor + '\n'
                               '            **kwargs\n'
                               '        )\n'
                               '    def check(self, obj, counter=None, **kwargs):\n'
                               '        return False  # TODO: implement the Dify loop condition\n'
        )

    def _gen_aggregator(self, nid: str, _step=None) -> str:
        data = self.g.data_of(nid)
        variables = data.get('variables') or []
        head = self._cls_head(nid, 'skeletons.Module')
        lines = ['    def on_process(self, obj, **kwargs):', '        values = []']
        for selector in variables:
            if isinstance(selector, dict):
                selector = selector.get('value_selector') or selector.get('variable_selector') or []
            if not selector:
                continue
            extra = ', '.join(repr(str(x)) for x in selector[1:])
            extra = f', {extra}' if extra else ''
            lines.append(f'        values.append(sel(obj, {str(selector[0])!r}{extra}))')
        lines += [
            '        output = next((v for v in values if v not in (None, \'\')), values[0] if values else \'\')',
            '        set_node(obj, self.node_id, output=output, result=output)',
            '        return obj',
        ]
        return head + '\n'.join(lines) + '\n'

    def _gen_assigner(self, nid: str, _step=None) -> str:
        data = self.g.data_of(nid)
        items = data.get('items') or data.get('variables') or []
        head = self._cls_head(nid, 'skeletons.Module')
        lines = ['    def on_process(self, obj, **kwargs):']
        if not items:
            lines.append('        return obj')
            return head + '\n'.join(lines) + '\n'
        for item in items:
            if not isinstance(item, dict):
                continue
            target = item.get('variable_selector') or item.get('assigned_variable_selector') or []
            source = item.get('value_selector') or item.get('input_selector') or []
            if not target:
                continue
            t_node, t_keys = str(target[0]), target[1:]
            key = t_keys[-1] if t_keys else 'value'
            if source:
                extra = ', '.join(repr(str(x)) for x in source[1:])
                extra = f', {extra}' if extra else ''
                lines.append(f'        _val = sel(obj, {str(source[0])!r}{extra})')
            else:
                lines.append(f'        _val = {item.get("value")!r}')
            lines.append(f'        set_node(obj, {t_node!r}, **{{{key!r}: _val}})')
            lines.append(f'        obj[{key!r}] = _val')
        lines.append('        set_node(obj, self.node_id)')
        lines.append('        return obj')
        return head + '\n'.join(lines) + '\n'

    def _gen_passthrough(self, nid: str, _step=None) -> str:
        head = self._cls_head(nid, 'skeletons.Module')
        return head + '    def on_process(self, obj, **kwargs):\n        return obj\n'

    def _gen_stub(self, nid: str, _step=None) -> str:
        ntype = self.g.type_of(nid)
        self.warnings.append(f'Unmapped node type `{ntype}`: {self.g.title_of(nid)} ({nid})')
        head = self._cls_head(nid, 'skeletons.Module')
        return head + (
            '    def on_process(self, obj, **kwargs):\n'
            f'        raise NotImplementedError({ntype!r} + " node is not auto-converted: " + {self.g.title_of(nid)!r})\n'
        )

    def _gen_model_class(self, tree: dict) -> str:
        ctor = self.emit_ctor(tree, indent=12)
        # Unwrap a top-level seq so Model(...) is not wrapped in an extra Pipeline.
        if tree.get('kind') == 'seq':
            inner = '\n'.join(self.emit_ctor(ch, indent=12) for ch in tree.get('children') or [])
        else:
            inner = ctor
        return f'''{self._register_deco()}
class Model(BaseModelWithoutDb):
    """{self.g.app_name}"""

    def __init__(self, cfgs={{}}, name=None, **kwargs):
        super().__init__(
{inner}
            name={self.g.app_name!r},
            **kwargs
        )
'''

    def _gen_main(self) -> str:
        start = self.g.start_id()
        kwargs_lines = []
        if start:
            for var in self.g.data_of(start).get('variables') or []:
                key = var.get('variable') or var.get('name')
                if not key:
                    continue
                if var.get('type') == 'number':
                    kwargs_lines.append(f'                {key!r}: 0,')
                else:
                    kwargs_lines.append(f'                {key!r}: \'\',')
        if not kwargs_lines:
            kwargs_lines.append('                # this workflow has no start variables')
        kwargs_block = '\n'.join(kwargs_lines)
        return f'''
if __name__ == '__main__':
    model = Model()
    result = model(
        {{
            'task_id': 'demo',
            'kwargs': {{
{kwargs_block}
            }},
        }},
        task_id='demo',
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
'''


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description='Convert a Dify workflow YAML into workflows modules')
    parser.add_argument('src', help='YAML file or directory')
    parser.add_argument('-o', '--out', default='dify_modules', help='output directory or file')
    parser.add_argument('--gen-test-code', action='store_true', help='append the if __name__ demo block')
    args = parser.parse_args(argv)

    def iter_yaml(path: Path) -> List[Path]:
        if path.is_file():
            return [path]
        files = sorted(path.glob('*.yml')) + sorted(path.glob('*.yaml'))
        return [p for p in files if p.is_file()]

    src = Path(args.src)
    files = iter_yaml(src)
    if not files:
        print(f'No YAML found: {src}', file=sys.stderr)
        return 1

    gen = CodeGenerator(files)

    out = Path(args.out)
    if len(files) == 1 and out.suffix == '.py':
        outputs = gen.to_files(out_files=[str(out)], gen_test_code=args.gen_test_code)
    else:
        outputs = gen.to_files(out_dir=str(out), gen_test_code=args.gen_test_code)

    for p in outputs:
        print(f'wrote {p}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
