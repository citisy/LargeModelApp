import uuid

from . import skeletons


class Visualizer:
    def str(self, module):
        if isinstance(module, skeletons.ModuleList):
            return self.str_module_list(module)

        else:
            return self.str_module(module)

    def str_module(self, module):
        if hasattr(module, 'config') and module.config:
            s = module.name + '(\n'
            for k, v in module.config.items():
                s += f'    {k}={v},\n'
            s += ')'
            return s
        else:
            return module.name

    def str_module_list(self, module_list):
        s = f'{self.str_module(module_list)}(\n'
        for _, module in module_list.modules:
            name = self.str(module)
            name = '\n'.join([' ' * 4 + _ for _ in name.split('\n')])
            s += name + '\n'
        s = s + ')'
        return s

    def module_info(self, module):
        if isinstance(module, skeletons.ModuleList):
            return self.module_info_module_list(module)

        else:
            return self.module_info_module(module)

    def module_info_module(self, module):
        return dict(
            name=module.name,
            comments=module.__doc__,
            apply=module.apply,
            mask=module.mask,
            allow_start=module.allow_start,
            allow_end=module.allow_end,
        )

    def module_info_module_list(self, module_list):
        s = []
        for name, module in module_list.modules:
            info = self.module_info(module)
            s.append(info)
        info = self.module_info_module(module_list)
        info['modules'] = s
        return info

    def flow_chat(self, module, *args, **kwargs):
        # download from https://graphviz.org/download/ first, and then,
        # pip install graphviz
        from graphviz import Digraph

        dot = Digraph()
        self._flow_chat(module, dot=dot)
        dot.render(*args, **kwargs)

    def _flow_chat(self, module, dot=None, last_names=None):
        from graphviz import Digraph

        if isinstance(module, skeletons.ModuleList):
            dot_name = f'cluster_{module.name}-{str(uuid.uuid4())[:8]}'
            sub_dot = Digraph(
                dot_name,
                graph_attr={
                    'label': module.name,
                    'fontname': "Helvetica,Arial,sans-serif",
                },
                node_attr={
                    'fontname': "Helvetica,Arial,sans-serif"
                },
                encoding='utf8'
            )
            if isinstance(module, skeletons.SwitchPipeline):
                cur_names, first_names = self.flow_chat_switch_pipeline(module, sub_dot)
            elif isinstance(module, skeletons.SkipPipeline):
                cur_names, first_names = self.flow_chat_skip_pipeline(module, sub_dot)
            elif isinstance(module, skeletons.LoopPipeline):
                cur_names, first_names = self.flow_chat_loop_pipeline(module, sub_dot)
            elif isinstance(module, skeletons.Pipeline):
                cur_names, first_names = self.flow_chat_pipeline(module, sub_dot)
            elif isinstance(module, skeletons.Sequential):
                cur_names, first_names = self.flow_chat_sequential(module, sub_dot)
            else:
                cur_names, first_names = self.flow_chat_module_list(module, sub_dot)
            dot.subgraph(sub_dot)

        else:
            cur_names, first_names = self.flow_chat_module(module, dot)

        if last_names:
            for a in last_names:
                for b in first_names:
                    dot.edge(a, b)
        return cur_names, first_names

    def flow_chat_module(self, module, dot, shape='box'):
        label = module.name
        if module.__doc__:
            label += '\n' + module.__doc__
        name = f'{module.name}-{str(uuid.uuid4())[:8]}'
        dot.node(name, label, shape=shape)

        cur_names = [name]

        return cur_names, cur_names

    def flow_chat_module_list(self, module_list, dot):
        first_name = f'{module_list.name}-{str(uuid.uuid4())[:8]}'
        dot.node(first_name, shape='point', style='filled', label='', height='.1', width='.1')

        last_name = f'{module_list.name}-{str(uuid.uuid4())[:8]}'
        dot.node(last_name, shape='point', style='filled', label='', height='.1', width='.1')

        for i, (name, module) in enumerate(module_list.modules):
            cur_names, _ = self.flow_chat_module(module, dot)
            cur_name = cur_names[0]

            dot.edge(first_name, cur_name)
            dot.edge(cur_name, last_name)

        last_names, first_names = [last_name], [first_name]
        return last_names, first_names

    def flow_chat_pipeline(self, module_list, dot):
        first_names = []
        last_names = []

        for i, (name, module) in enumerate(module_list.modules):
            last_names, _first_names = self._flow_chat(module, dot, last_names)

            if i == 0:
                first_names = _first_names

        return last_names, first_names

    def flow_chat_sequential(self, module_list, dot):
        first_names = []
        last_names = []
        output_names = []

        for i, (name, module) in enumerate(module_list.modules):
            if isinstance(module, skeletons.BaseSequentialInput):
                last_names, _first_names = self.flow_chat_module(module, dot, 'invtrapezium')
            elif isinstance(module, skeletons.BaseSequentialOutput):
                output_names, _first_names = self.flow_chat_module(module, dot, 'trapezium')
            else:
                last_names, _first_names = self._flow_chat(module, dot, last_names)

            if i == 0:
                first_names = _first_names

        for a in last_names:
            for b in output_names:
                dot.edge(a, b)
        last_names = output_names
        return last_names, first_names

    def flow_chat_switch_pipeline(self, module_list, dot):
        node_name = f'{module_list.name}-{str(uuid.uuid4())[:8]}'
        dot.node(node_name, 'Switch', shape='invhouse')
        _last_names = [node_name]
        first_names = [node_name]

        last_names = []
        for i, (name, module) in enumerate(module_list.modules):
            cur_names, _first_names = self._flow_chat(module, dot, _last_names)
            last_names += cur_names

        return last_names, first_names

    def flow_chat_skip_pipeline(self, module_list, dot):
        node_name = f'{module_list.name}-{str(uuid.uuid4())[:8]}'
        dot.node(node_name, 'Skip?', shape='diamond')
        last_names = [node_name]
        first_names = [node_name]

        for i, (name, module) in enumerate(module_list.modules):
            last_names, _first_names = self._flow_chat(module, dot, last_names)

        last_names = [node_name] + last_names
        return last_names, first_names

    def flow_chat_loop_pipeline(self, module_list: skeletons.LoopPipeline, dot):
        node_name = f'{module_list.name}-{str(uuid.uuid4())[:8]}'
        dot.node(node_name, 'Check', shape='diamond')

        first_names = []
        last_names = []
        if module_list.check_before_loop:
            last_names = [node_name]

        for i, (name, module) in enumerate(module_list.modules):
            last_names, _first_names = self._flow_chat(module, dot, last_names)

            if i == 0:
                first_names = _first_names

        if module_list.check_before_loop:
            first_names = [node_name]
            for a in last_names:
                dot.edge(a, node_name)
        else:
            for a in last_names:
                dot.edge(a, node_name)

            for a in first_names:
                dot.edge(node_name, a)

            last_names = [node_name]

        return last_names, first_names
