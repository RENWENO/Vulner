#! -*- coding: utf-8 -*-
# @Time    : 2025/4/2 13:28
# @Author  : xx
import os
import re
import json
import networkx as nx
from pathlib import Path
from solcx import compile_files, install_solc, get_installable_solc_versions
from typing import Dict, List, Optional
import tempfile

#print(get_installable_solc_versions())
class SolidityASTConverter:
    def __init__(self):
        self.installed_versions = [str(v) for v in get_installable_solc_versions()]
        self.graph = nx.DiGraph()
        self._node_counter = 0
        self._current_file = ""

    def convert_file(self, sol_path: Path) -> nx.DiGraph:
        """转换单个.sol文件"""
        # 读取Solidity文件
        with open(sol_path, "r") as f:
            code = f.read()

        # 自动检测版本
        version = self._detect_version(code)
        if not version:
            raise ValueError("无法自动检测Solidity版本")

        # 安装必要编译器
        if version not in self.installed_versions:
            install_solc(version)
            self.installed_versions.append(version)

        # 编译并获取AST
        ast = self._compile_to_ast(sol_path, version)

        # 转换为NetworkX图
        G = self._convert_ast_to_nx(ast)

        # 保存结果


        return G
        # try:
        #
        #
        # except Exception as e:
        #     print(f"转换失败: {str(e)}")
        #     return nx.DiGraph()

    def _detect_version(self, code: str) -> Optional[str]:
        """智能版本检测"""
        version_pattern = r'pragma\s+solidity\s+([\^><=]*\s*[\d\.]+)'
        #print(version_pattern)
        match = re.search(version_pattern, code)
        if not match:
            return None

        version_spec = match.group(1)
        base_version = re.sub(r'[\^><=~]', '', version_spec).strip()
        #print(base_version)
        return self._find_closest_version(base_version)

    def _find_closest_version(self, base: str) -> str:
        """版本匹配策略"""
        base_major, base_minor, _ = base.split('.')
        for v in sorted(self.installed_versions, reverse=True):
            major, minor, _ = v.split('.')
            if major == base_major and minor == base_minor:
                if base == '0.5.0':
                    return base
                if base == '0.4.25':
                    return base
                if base == '0.5.8':
                    return base
                if base == '0.5.2':
                    return base
                if base == '0.5.1':
                    return base
                if base == '0.4.24':
                    return  base
                if base == '0.4.20':
                    return  base
                if base == '0.4.21':
                    return  base
                if base == '0.4.19':
                    return base
                if base == '0.4.14':
                    return base

                if base == '0.5.7':
                    return base
                return v
        return self.installed_versions[-1] if self.installed_versions else None

    def _compile_to_ast(self, sol_path: Path, version: str) -> Dict:
        """编译获取AST"""
        #print(version)
        compiled = compile_files(
            [str(sol_path)],
            output_values=["ast"],
            solc_version=version,
            import_remappings=[
                f"@openzeppelin/=node_modules/@openzeppelin/",
                f"@/=/{sol_path.parent}/"
            ]
        )
        # print(sol_path.name)
        # aa = str(sol_path)+":Registry"
        aa = list(compiled.keys())[0]
        return compiled[aa]["ast"]

    def _convert_ast_to_nx(self, ast: Dict):
        """递归转换AST结构"""

        G = nx.DiGraph()

        def process_node(node, parent_id=None):
            # 提取节点信息
            node_id = node.get('id', None)
            if node_id is None:
                return  # 忽略无id的节点

            # 提取属性
            attributes = node.get('attributes', {})
            node_data = {
                'name': node.get('name', ''),
                'src': node.get('src', ''),
                **attributes
            }

            # 添加节点
            G.add_node(node_id, **node_data)

            # 添加父节点到当前节点的边
            if parent_id is not None:
                G.add_edge(parent_id, node_id)

            # 递归处理子节点
            children = node.get('children', [])
            for child in children:
                process_node(child, parent_id=node_id)

        # 从根节点开始处理
        process_node(ast)
        return G


# 使用示例
if __name__ == "__main__":
    # 初始化转换器
    converter = SolidityASTConverter()
    # String = "./sol_149363/" + "0x0000000f8eF4be2B7AeD6724e893c1b674B9682D.sol"
    # sample_contract = Path(String)
    # print(sample_contract)
    # output_dir = Path("ast_graphs")
    # output_dir.mkdir(exist_ok=True)
    #
    # #执行转换
    # graph = converter.convert_file(sample_contract)

    #转换单个文件
    namelist = os.listdir("./0424File/")
    print(len(namelist))
    for namedir in namelist:
        try:
            String = "./0424File/"+namedir
            sample_contract = Path(String)
            #print(sample_contract)
            output_dir = Path("ast_graphs")
            output_dir.mkdir(exist_ok=True)

            # 执行转换
            graph = converter.convert_file(sample_contract)
            print(f"节点总数: {graph.number_of_nodes()}")
            print(f"边总数: {graph.number_of_edges()}")
        except Exception as e:
            print(e)
            string="/home/Vulner/0424File/"+namedir
            #os.remove(string)
            print(string)
            continue

        # 基础分析
        # print(f"节点总数: {graph.number_of_nodes()}")
        # print(f"边总数: {graph.number_of_edges()}")
        # print("前5个节点属性示例:")
        # for node in list(graph.nodes(data=True))[:90]:
        #     print(node)

    print(len(os.listdir("./0424File")))
    # 可视化
    #converter.visualize()