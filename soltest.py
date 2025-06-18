#! -*- coding: utf-8 -*-
# @Time    : 2025/4/8 0:34
# @Author  : xx
import re
import os
import subprocess
import argparse
from pathlib import Path
from packaging.specifiers import SpecifierSet
from packaging.version import Version
import shutil


def extract_version_constraint(file_path: Path) -> str:
    """
    提取 Solidity 文件的版本约束（如 ^0.8.0）
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 使用正则匹配 pragma 声明
    pragma_match = re.search(r'pragma\s+solidity\s+([^;]+);', content)
    if not pragma_match:
        raise ValueError(f"No pragma found in {file_path}")

    constraint = pragma_match.group(1).strip()
    return constraint



def find_closest_version(base: str) -> str:
    """版本匹配策略"""
    installed_versions = get_installed_solc_versions()
    base_major, base_minor, _ = base.split('.')
    for v in sorted(installed_versions, reverse=True):
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
                return base
            if base == '0.4.20':
                return base
            if base == '0.4.21':
                return base
            if base == '0.4.19':
                return base
            if base == '0.4.14':
                return base

            if base == '0.5.7':
                return base
            if base == '0.4.11':
                return base
            return v
    return installed_versions[-1] if installed_versions else None


def get_installed_solc_versions() -> list:
    """
    获取已安装的 solc 版本列表
    """
    result = subprocess.run(
        ['/usr/local/miniconda3/bin/solc-select', 'versions'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    versions = []
    for line in result.stdout.split('\n'):
        version_match = re.search(r'(\d+\.\d+\.\d+)', line)
        if version_match:
            versions.append(version_match.group(1))
    return versions


def move_files_by_extension(src_dir, dst_dir, extensions, overwrite=False):
    """
    移动指定后缀的文件到目标文件夹
    :param src_dir: 源文件夹路径
    :param dst_dir: 目标文件夹路径
    :param extensions: 文件后缀列表（例如 ['.dot', '.txt']）
    :param overwrite: 是否覆盖目标文件夹中的同名文件
    :return: 移动成功的文件列表
    """
    moved_files = []

    try:
        # 验证源文件夹是否存在
        if not os.path.exists(src_dir):
            raise FileNotFoundError(f"源文件夹 {src_dir} 不存在")

        # 创建目标文件夹（如果不存在）
        os.makedirs(dst_dir, exist_ok=True)

        # 遍历源文件夹
        for filename in os.listdir(src_dir):
            # 检查文件后缀
            if any(filename.lower().endswith(ext) for ext in extensions):
                src_path = os.path.join(src_dir, filename)
                dst_path = os.path.join(dst_dir, filename)

                # 处理同名文件
                if os.path.exists(dst_path):
                    if overwrite:
                        os.remove(dst_path)
                    else:
                        #print(f"跳过已存在文件: {filename}")
                        continue

                # 移动文件
                shutil.move(src_path, dst_path)
                moved_files.append(filename)
                #print(f"已移动: {filename}")

        #print(f"\n操作完成！共移动 {len(moved_files)} 个文件到 {dst_dir}")
        return moved_files

    except Exception as e:
        print(f"操作失败: {str(e)}")
        return []


def generate_cfg(sol_file: Path, output_dir: Path, solc_version: str):
    """
    使用 Slither 生成 CFG 图
    """
    name = str(sol_file).split('/')[-1][:-4]
    ####修改源码路径
    Srcdic = '/home/Vulner/0424_Dos_nobug_sol'
    extensions = ['.dot']
    #print(name)
    output_dir=output_dir+'/'+name
    output_dir = Path(output_dir)
    # 创建输出目录
    #output_dir.mkdir(parents=True, exist_ok=True)
    # 设置 Solidity 版本
    result1 = subprocess.run(
        ['/usr/local/miniconda3/bin/solc-select', 'use', solc_version],
        check=True,
        capture_output=True,
        text=True
    )
    #print(result1)


    # 运行 Slither 生成 CFG
    #cmd = f"slither {sol_file} --print cfg --output {output_dir}"

    #command = ['/usr/local/miniconda3/bin/slither', sol_file, '--print', 'cfg']
    command = ['slither', sol_file, '--print', 'cfg']
    subprocess.run(
    command,
    capture_output=True,
    text=True,
    check=True)

    move_files_by_extension(Srcdic,output_dir,extensions,)
    return True

if __name__ == "__main__":
    # 提取版本约束
    ###修改控制流图输出路径
    output_path = './CFG/DoS_nobug_CFG'
    ####修改源码路径
    namelist = os.listdir("./0424_Dos_nobug_sol")
    print(len(namelist))
    for namedir in namelist:
        try:
            ###修改源码路径
            sol_file = '/home/Vulner/0424_Dos_nobug_sol/'+namedir
            raw_constraint = extract_version_constraint(sol_file)
            #solc_version = find_closest_version(raw_constraint)
            solc_version = '0.4.24'

            # 匹配最佳 solc 版本
            #solc_version = find_compatible_solc_version(constraint)
            # 生成 CFG
            out = generate_cfg(sol_file, output_path , solc_version)
            #print(out)
            print(f"[+] Generated CFG for {sol_file} using solc {solc_version}")
        except Exception as e:
            print("errr")
            print(e)

            continue

