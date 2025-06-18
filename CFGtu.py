#! -*- coding: utf-8 -*-
# @Time    : 2025/4/5 17:43
# @Author  : xx
import json
import networkx as nx
import subprocess
from typing import Dict, List, Optional
#from graphviz import Digraph


class CFGBuilder:
    """控制流图构建器"""

    def __init__(self):
        self.cfg = nx.DiGraph()  # 使用NetworkX存储CFG
        self.current_block = None  # 当前基本块
        self.block_stack = []  # 块栈用于处理嵌套结构
        self.block_counter = 0  # 基本块计数器
        self.function_entry = {}  # 函数入口映射表

    class BasicBlock:
        """基本块数据结构"""

        def __init__(self, bid):
            self.id = bid  # 块ID
            self.statements = []  # 包含的语句
            self.next_block = None  # 直接后继
            self.branch_blocks = {}  # 条件分支（条件 -> 目标块）

    def _new_block(self) -> BasicBlock:
        """创建新的基本块"""
        self.block_counter += 1
        new_block = self.BasicBlock(self.block_counter)
        self.cfg.add_node(new_block.id, block=new_block)
        return new_block

    def _add_edge(self, source: BasicBlock, target: BasicBlock, label: str = None):
        """添加控制流边"""
        self.cfg.add_edge(source.id, target.id, label=label)

    def _process_if_statement(self, node: Dict):
        """处理if语句结构"""
        # 创建条件块
        condition_block = self.current_block
        condition_block.statements.append(node['condition'])

        # 创建true分支
        true_block = self._new_block()
        self._add_edge(condition_block, true_block, "true")

        # 创建false分支（如果有else）
        false_block = self._new_block()
        self._add_edge(condition_block, false_block, "false")

        # 处理true分支体
        self.block_stack.append(self.current_block)
        self.current_block = true_block
        self._process_statement(node['TrueBody'])
        true_exit = self.current_block

        # 处理false分支体
        if 'FalseBody' in node:
            self.current_block = false_block
            self._process_statement(node['FalseBody'])
            false_exit = self.current_block
        else:
            false_exit = false_block

        # 创建合并块
        merge_block = self._new_block()
        self._add_edge(true_exit, merge_block)
        self._add_edge(false_exit, merge_block)

        # 恢复当前块
        self.current_block = merge_block
        self.block_stack.pop()

    def _process_loop(self, node: Dict, loop_type: str):
        """通用循环处理（支持for/while）"""
        # 初始化块
        init_block = self.current_block

        # 条件块
        cond_block = self._new_block()
        self._add_edge(init_block, cond_block)

        # 循环体块
        body_block = self._new_block()
        self._add_edge(cond_block, body_block, "true")

        # 退出块
        exit_block = self._new_block()
        self._add_edge(cond_block, exit_block, "false")

        # 处理循环体
        self.block_stack.append(self.current_block)
        self.current_block = body_block
        self._process_statement(node['body'])
        loop_exit = self.current_block

        # 后置处理（仅for循环）
        if loop_type == 'ForStatement' and 'post' in node:
            post_block = self._new_block()
            self._add_edge(loop_exit, post_block)
            self._add_edge(post_block, cond_block)
        else:
            self._add_edge(loop_exit, cond_block)

        # 恢复当前块
        self.current_block = exit_block
        self.block_stack.pop()

    def _process_statement(self, node: Dict):
        """递归处理AST节点"""
        if isinstance(node, list):
            for stmt in node:
                self._process_statement(stmt)
            return

        node_type = node.get('nodeType')

        # 普通语句
        if node_type in ['ExpressionStatement', 'VariableDeclaration']:
            self.current_block.statements.append(node)

        # 控制流结构
        elif node_type == 'IfStatement':
            self._process_if_statement(node)
        elif node_type in ['ForStatement', 'WhileStatement']:
            self._process_loop(node, node_type)
        elif node_type == 'ReturnStatement':
            self.current_block.statements.append(node)
            self.current_block.next_block = None
        elif node_type == 'FunctionDefinition':
            self._process_function(node)

        # 处理子节点
        for key in ['body', 'statements', 'expression']:
            if key in node:
                self._process_statement(node[key])

    def _process_function(self, node: Dict):
        """处理函数定义"""
        # 创建入口块
        entry_block = self._new_block()
        self.function_entry[node['name']] = entry_block
        self.current_block = entry_block

        # 处理参数和返回值
        if 'parameters' in node:
            self.current_block.statements.append(node['parameters'])
        if 'returnParameters' in node:
            self.current_block.statements.append(node['returnParameters'])

        # 处理函数体
        if 'body' in node:
            self._process_statement(node['body'])

        # 添加隐式返回（如果没有显式返回）
        if self.current_block.next_block is None:
            exit_block = self._new_block()
            self._add_edge(self.current_block, exit_block)

    def build(self, ast: Dict) -> nx.DiGraph:
        """构建控制流图主入口"""
        for source in ast['sources'].values():
            for contract in source['ast']['nodes']:
                if contract['nodeType'] == 'ContractDefinition':
                    for node in contract['nodes']:
                        if node['nodeType'] == 'FunctionDefinition':
                            self._process_function(node)
        return self.cfg




# 使用示例
if __name__ == "__main__":
    # 生成AST
    def get_ast(sol_file: str) -> Dict:
        result = subprocess.run(
            ['solc', '--ast-json', sol_file],
            capture_output=True, text=True
        )
        return json.loads(result.stdout)


    # 测试合约
    sample_contract = """
    pragma solidity ^0.8.0;

    contract CFGExample {
        function test(uint x) public pure returns (uint) {
            uint y = 0;
            if (x > 10) {
                for (uint i=0; i<5; i++) {
                    y += 1;
                }
            } else {
                y = x;
            }
            return y;
        }
    }
    """

    # 写入临时文件
    with open('temp.sol', 'w') as f:
        f.write(sample_contract)

    try:
        # 构建CFG
        builder = CFGBuilder()
        ast = get_ast('temp.sol')
        cfg = builder.build(ast)

        # 输出统计信息
        print(f"CFG包含 {len(cfg.nodes)} 个基本块和 {len(cfg.edges)} 条边")

        # 可视化
        builder.visualize('sample_cfg')

    finally:
        import os

        os.remove('temp.sol')