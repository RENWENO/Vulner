#! -*- coding: utf-8 -*-
# @Time    : 2025/4/2 23:59
# @Author  : xx
import json
import networkx as nx
from pathlib import Path
from solcx import compile_files, install_solc, get_installable_solc_versions
from typing import Dict, Any, List
import logging
import re
# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AST-Converter")


class EnhancedASTConverter:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_map = {}  # 用于跟踪原始AST节点到图节点的映射
        self._current_id = 0

    def convert(self, sol_path: Path, solc_version: str = None) -> nx.DiGraph:
        """主转换方法"""
        try:
            # 验证文件存在性
            if not sol_path.exists():
                raise FileNotFoundError(f"文件 {sol_path} 不存在")

            # 安装必要编译器版本
            self._setup_compiler(sol_path, solc_version)

            # 编译并获取AST
            ast = self._compile_with_debug(sol_path)

            # 转换AST为图结构
            self._convert_ast(ast)

            return self.graph

        except Exception as e:
            logger.error(f"转换失败: {str(e)}")
            return nx.DiGraph()

    def _setup_compiler(self, sol_path: Path, version: str):
        """编译器版本管理"""
        with open(sol_path, 'r') as f:
            code = f.read()

        # 自动检测版本
        detected_version = self._detect_solidity_version(code) if not version else version
        if not detected_version:
            raise ValueError("无法自动检测Solidity版本")

        # 安装编译器
        if detected_version not in get_installable_solc_versions():
            logger.info(f"正在安装Solidity编译器 {detected_version}")
            install_solc(detected_version)

    def _detect_solidity_version(self, code: str) -> str:
        """改进的版本检测"""
        version_patterns = [
            r'pragma\s+solidity\s+([\^><=]*\s*[\d\.]+)\s*;',
            r'@solidity\s+([\d\.]+)'
        ]
        for pattern in version_patterns:
            match = re.search(pattern, code)
            if match:
                version = re.sub(r'[\^><=~]', '', match.group(1)).strip()
                return self._resolve_version(version)
        return None

    def _resolve_version(self, base_version: str) -> str:
        """版本解析策略"""
        available_versions = sorted(get_installable_solc_versions(), reverse=True)
        base_major, base_minor, _ = base_version.split('.')
        for ver in available_versions:
            major, minor, _ = str(ver).split('.')
            if major == base_major and minor == base_minor:
                return str(ver)
        return str(available_versions[0]) if available_versions else None

    def _compile_with_debug(self, sol_path: Path) -> Dict:
        """带调试信息的编译方法"""
        compile_args = {
            "import_remappings": [
                f"@openzeppelin/=node_modules/@openzeppelin/",
                f"@/=/{sol_path.parent}/"
            ],
            "output_values": ["ast"],
            "allow_paths": [str(sol_path.parent)]
        }

        logger.debug(f"编译参数: {json.dumps(compile_args, indent=2)}")

        try:
            compiled = compile_files([str(sol_path)], **compile_args)
            print("开始提取")
            logger.debug("原始AST结构示例:\n" + json.dumps(
                list(compiled.values())[0]['ast'], indent=2)[:1000] + "...")
            return compiled[sol_path.name]
        except Exception as e:
            logger.error(f"编译错误: {str(e)}")
            raise

    def _convert_ast(self, ast: Dict):
        """增强的AST转换方法"""

        def recursive_parse(node: Dict, parent_id: int = None):
            # 创建当前节点
            current_id = self._create_node(node)

            # 记录父子关系
            if parent_id is not None:
                self.graph.add_edge(parent_id, current_id)
                logger.debug(f"添加边: {parent_id} -> {current_id}")

            # 处理所有可能包含子节点的字段
            child_fields = self._find_child_fields(node)
            for field in child_fields:
                for child in node[field]:
                    if isinstance(child, dict):
                        recursive_parse(child, current_id)
                    elif isinstance(child, list):
                        for subchild in child:
                            if isinstance(subchild, dict):
                                recursive_parse(subchild, current_id)

        # 开始递归解析
        recursive_parse(ast)

    def _create_node(self, node: Dict) -> int:
        """创建图节点并返回ID"""
        node_hash = hash(json.dumps(node, sort_keys=True))
        if node_hash in self.node_map:
            return self.node_map[node_hash]

        node_id = self._current_id
        self._current_id += 1

        # 提取关键属性
        attributes = {
            'node_type': node.get('nodeType', 'Unknown'),
            'src': node.get('src', ''),
            'name': node.get('name', ''),
            'type': self._get_type_info(node),
            'children': []
        }

        # 添加特殊属性
        if 'value' in node:
            attributes['value'] = node['value']
        if 'literals' in node:
            attributes['literals'] = node['literals']

        self.graph.add_node(node_id, **attributes)
        self.node_map[node_hash] = node_id

        logger.debug(f"创建节点 {node_id}: {attributes}")
        return node_id

    def _get_type_info(self, node: Dict) -> str:
        """提取类型信息"""
        type_mapping = {
            'VariableDeclaration': node.get('typeName', {}).get('name', ''),
            'FunctionDefinition': node.get('visibility', '') + ' ' + node.get('stateMutability', ''),
            'ExpressionStatement': node.get('expression', {}).get('nodeType', ''),
        }
        return type_mapping.get(node.get('nodeType'), node.get('typeDescriptions', {}).get('typeString', ''))

    def _find_child_fields(self, node: Dict) -> List[str]:
        """自动发现包含子节点的字段"""
        return [key for key, value in node.items()
                if isinstance(value, (list, dict)) and key not in ['src', 'typeDescriptions']]

    def save_graph(self, output_path: Path):
        """保存图结构"""
        nx.write_graphml(self.graph, output_path)
        logger.info(f"图结构已保存至: {output_path}")

    def visualize_subgraph(self, max_nodes=50):
        """可视化子图（防止节点过多卡顿）"""
        import matplotlib.pyplot as plt

        if len(self.graph.nodes) > max_nodes:
            logger.warning(f"图过大（{len(self.graph.nodes)}节点），仅显示前{max_nodes}个节点")
            subgraph = self.graph.subgraph(list(self.graph.nodes)[:max_nodes])
        else:
            subgraph = self.graph

        plt.figure(figsize=(20, 15))
        pos = nx.kamada_kawai_layout(subgraph)
        nx.draw(subgraph, pos, with_labels=True,
                labels=nx.get_node_attributes(subgraph, 'node_type'))
        plt.show()


# 使用示例
if __name__ == "__main__":
    converter = EnhancedASTConverter()

    # 输入文件路径
    contract_path = Path("./Sample.sol")

    # 执行转换
    ast_graph = converter.convert(contract_path)

    # 输出统计信息
    print(f"转换结果包含 {len(ast_graph.nodes)} 个节点和 {len(ast_graph.edges)} 条边")

    # 保存结果
    output_path = Path("ast_graphs") / f"{contract_path.stem}.graphml"
    converter.save_graph(output_path)

    # 可视化子图
    converter.visualize_subgraph()
