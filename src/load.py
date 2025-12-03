import os

import pandas as pd
import pytest
import logging
import ast
import random
from _ast import AST
from enum import Enum
from pathlib import Path
from typing import List, Optional
from py2cfg import CFGBuilder, CFG

class Language(Enum):
    PYTHON = 1
    PYTHON_TEST_SUITE = 2
    JAVA = 3

class Program:
    def __init__(self, path: Optional[Path] = None, source: Optional[str] = None):
        self.results = None
        self.path_tests = None
        self.source_tests = None
        self.testcases = None
        self.source = source
        self.path = path
        self.subsets = []

        if path is not None and source is None:
            self.from_file(str(path))
        elif source is not None and path is None:
            self.from_source(source)

    def from_file(self, path: str) -> str:
        p = Path(path)
        source = p.read_text(encoding="utf-8")
        self.source = source
        self.path = p
        return source

    def from_source(self, source: str) -> str:
        self.source = source
        return source

    def dump(self) -> str:
        return self.source

    def test_dataset(self, testcases_path: str) -> str:
        """Return a list of booleans indicating whether the set of tests for the program fails or passes."""
        self.testcases = []
        assert self.path is not None
        test_file = Path(testcases_path)
        self.path_tests = testcases_path
        self.source_tests = test_file.read_text(encoding="utf-8")
        return self.source_tests

    def generate_subsets(self, subset_size: int, overlapping: bool = False) -> List[List[str]]:
        return self.subsets

    def run_tests(self) -> List[bool]:
        if self.source_tests is None:
            raise ValueError("Test cases not loaded. Please run test_dataset first.")

        # use os.system to run the tests
        results = [] # list of dicts with tests results and reasons
        test_file_path = self.path_tests
        if test_file_path is None:
            raise ValueError("Test cases path is not set.")

        os.system(f"pytest {test_file_path} --tb=short --no-header -vv >> pytest_output.txt")
        with open("pytest_output.txt", "r") as f:
            lines = f.readlines()
            complete_output = "".join(lines)
            start_index = 0
            for i, line in enumerate(lines):
                if line.startswith("collecting"):
                    start_index = i + 1
            for line in lines[start_index:]:
                if line.find("FAILED") != -1:
                    reason = complete_output.find(f"FAILED {line.split(" ")[0]}::")
                    reason = complete_output[reason:reason+5].strip()
                    print(f"Reason: {reason} at line: {line}")
                    results.append({
                        "failed": True,
                        "test": line.split(" ")[0],
                        "reason": reason
                    })
                elif line.find("PASSED") != -1:
                    results.append({
                        "failed": False,
                        "test": line.split(" ")[0],
                        "reason": None
                    })
                elif line.startswith("==="):
                    break
        os.remove("pytest_output.txt")
        self.results = results
        return results

    def rerun_tests(self, failed_only: bool = True) -> Optional[List[dict]]:
        if self.results is None:
            self.run_tests()

        df = pd.DataFrame(self.results)
        failed_tests = df[df["failed"] == failed_only]
        if len(failed_tests) > 0:
            print(f"Number of failed tests: {len(failed_tests)}")
            new_results = []
            for _, row in failed_tests.iterrows():
                print(f"Test: {row['test']}, Reason: {row['reason']}")
                # rerun
                os.system(f"pytest row['test'] --tb=short --no-header -vv >> pytest_output.txt")
                # collect results:
                with open("pytest_output.txt", "r") as f:
                    lines = f.read()
                    failed = lines.find("FAILED") != -1
                    if failed:
                        reason = lines.find(f"FAILED {row['test']}")
                        reason = lines[reason:reason + 5].strip()
                        new_results.append({
                            "failed": True,
                            "test": row['test'],
                            "reason": reason
                        })
                    else:
                        new_results.append({
                            "failed": False,
                            "test": row['test'],
                            "reason": None
                        })
                os.remove("pytest_output.txt")
            print("Rerun results:")
            for res in new_results:
                print(f"Test: {res['test']}, Failed: {res['failed']}, Reason: {res['reason']}")
            return new_results
        else:
            print("No failed tests found.")
            return None


class LinesProgram(Program):
    def __init__(self, lines: Optional[list[str]] = None, source: str = None, path: Optional[Path] = None):
        super().__init__(path, source)
        self.path = path
        self.source = source
        self.lines = lines
        self.subsets = []

        if path is not None and source is None:
            self.from_file(str(path))
        elif source is not None and path is None:
            self.from_source(source)

    def from_file(self, path: str) -> list[str]:
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        self.lines = text.splitlines()
        self.path = p
        self.source = text
        return self.lines

    def from_source(self, source: str) -> list[str]:
        self.lines = source.splitlines()
        self.source = source
        return self.lines

    def text(self) -> str:
        return "\n".join(self.lines)

    def get_line(self, index: int) -> str:
        return self.lines[index]

    def generate_subsets(self, subset_size: int, overlapping: bool = False) -> List[List[str]]:
        subsets = []
        total_lines = len(self.lines)
        if not overlapping:
            # makes sure subset size is not larger than total lines
            if subset_size > total_lines:
                subset_size = total_lines
            for i in range(0, total_lines, subset_size):
                subset = self.lines[i : i + subset_size]
                subsets.append(subset)
            return subsets
        else:
            # randomly select non-overlapping subsets
            indices = list(range(total_lines - subset_size + 1))
            random.shuffle(indices)
            selected_indices = indices[: (total_lines // subset_size)]
            for idx in selected_indices:
                subset = self.lines[idx : idx + subset_size]
                subsets.append(subset)
            return subsets

class ASTProgram(Program):
    def __init__(self, tree: Optional[ast.Module] = None, path: Optional[Path] = None, source: Optional[str] = None):
        super().__init__(path)
        self.tree = tree
        self.path = path
        self.source = source
        self.subsets = []

        if path is not None and source is None:
            self.from_file(str(path))
        elif source is not None and path is None:
            self.from_source(source)

    def from_file(self, path: str) -> ast.Module:
        p = Path(path)
        source = p.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(p))
        self.source = source
        self.path = p
        self.tree = tree
        return tree

    def from_source(self, source: str) -> ast.Module:
        tree = ast.parse(source)
        self.source = source
        self.tree = tree
        return tree

    def test_dataset(self, testcases_path: str) -> ast.Module:
        self.testcases = []
        assert self.path is not None
        test_file = Path(testcases_path)
        self.path_tests = testcases_path
        self.source_tests = test_file.read_text(encoding="utf-8")
        test_tree = ast.parse(self.source_tests)
        self.testcases = test_tree
        return self.testcases

    def collect_nodes(self) -> List[ast.AST]:
        nodes: List[ast.AST] = []

        class Visitor(ast.NodeVisitor):
            def visit(self, node: ast.AST) -> None:
                nodes.append(node)
                self.generic_visit(node)

        Visitor().visit(self.tree)
        return nodes

    def dump(self, **kwargs) -> str:
        return ast.dump(self.tree, **kwargs)

class CFGProgram(Program):
    def __init__(self, cfg: Optional[CFG], path: Optional[Path] = None, source: Optional[str] = None):
        super().__init__(path, source)
        self.cfg = cfg
        self.path = path
        self.source = source
        self.subsets = []

    def from_file(self, path: str) -> CFG:
        cfg =  CFGBuilder().build_from_file("file", path)
        p = Path(path)
        source = p.read_text(encoding="utf-8")
        self.source = source
        self.path = p
        self.cfg = cfg
        return cfg

    def from_source(self, source: str) -> CFG:
        cfg = CFGBuilder().build_from_src("source", source)
        self.source = source
        self.cfg = cfg
        return cfg

    def dump(self) -> str:
        return str(self.cfg)

    def draw(self, output_path: str) -> None:
        p = Path(output_path)
        self.cfg.build_visual(output_path, format=p.suffix)

    def get_nodes(self):
        return self.cfg.own_blocks()


class Block:
    """
    Generic representation of a block in a Program.
    - Can be a single line or a group of lines.
    - Can be a subtree or a node in the AST.
    - Can be a CFG block or a CFG node.

    Contains references to the original source code and its location through the Program object.
    """

    def __init__(self, program: Program|LinesProgram|ASTProgram|CFGProgram, start_index: int = None, end_index: int = None):
        self.program = program
        # Indexes with reference to a list of lines/nodes in the program
        self.start_index = start_index
        self.end_index = end_index
        if type(program) == LinesProgram:
            self.kind = "lines"
        elif type(program) == ASTProgram:
            self.kind = "ast"
        elif type(program) == CFGProgram:
            self.kind = "cfg"
        else:
            self.kind = "unknown"

class ASTBlock(Block):
    def __init__(self, program: ASTProgram, start_index: int = None, end_index: int = None):
        super().__init__(program, start_index, end_index)
        self.kind = "ast"
        assert type(program) == ASTProgram

    def __str__(self):
        return f"ASTBlock(start_index={self.start_index}, end_index={self.end_index})"

    def display(self):
        """Display the AST block."""
        subtree = self.get_subtree()
        for node in subtree:
            print(ast.dump(node, indent=4))


    def get_subtree(self) -> list[AST]:
        """Return a list of AST nodes that make up the subtree."""
        nodes = self.program.collect_nodes()
        if self.start_index is not None and self.end_index is not None:
            # use the original tree nodes to create a new subtree
            start_node = nodes[self.start_index]
            end_node = nodes[self.end_index]
            if start_node is end_node:
                return [start_node]
            else:
                return nodes[self.start_index : self.end_index + 1]
        else:
            return nodes



def convert_file_to_ast(file_path: str, language: Language):
    logging.info(f"Converting file {file_path} to ast in {language}")
    if (language == Language.PYTHON or language == Language.PYTHON_TEST_SUITE) and os.path.isfile(file_path) and file_path.endswith(".py"):
        # using the file path to get the ast
        print(file_path)
        return ast.parse(open(file_path).read())
    else:
        return None # for now


def load_file(file_path: str, language: Language) -> ast.Module:
    return convert_file_to_ast(file_path, language)

def collect_all_nodes(tree):
    """Collect all AST nodes with their parent-child relationships."""
    nodes = []

    class Visitor(ast.NodeVisitor):
        def visit(self, node):
            nodes.append(node)
            self.generic_visit(node)

    Visitor().visit(tree)
    return nodes

def is_ancestor(node1, node2):
    """Check if node1 is an ancestor of node2."""
    for child in ast.walk(node1):
        if child is node2:
            return True
    return False

def get_non_overlapping_subtrees(all_nodes, count=4):
    """Randomly select non-overlapping subtrees."""
    selected = []
    number_of_nodes = len(all_nodes)
    tries = 0
    max_tries = 1000

    for _ in range(count):
        tries += 1
        if tries > max_tries:
            raise ValueError("Could not find enough non-overlapping subtrees")
        first_node = random.choice(all_nodes)
        # create a tree untill the first node is found
        tree = ast.Module(body=[first_node])
        for node in all_nodes:
            if node is first_node:
                continue
            if not is_ancestor(tree, node):
                tree.body.append(node)
        selected.append(tree)
        print(ast.dump(tree, indent=4))
        # all_nodes.remove()


    if len(selected) < count: 
        raise ValueError("Could not find enough non-overlapping subtrees")

    return selected

if __name__ == "__main__":
    # visualise tree
    tree = load_file("../examples/test_equl.py", Language.PYTHON)
    print(ast.dump(tree, indent=4))

    all_nodes = collect_all_nodes(tree)

    # Select 4 random, non-overlapping subtrees
    subs = get_non_overlapping_subtrees(all_nodes, count=4)

    # Print the selected subtrees
    for i, sub in enumerate(subs):
        print(f"\nSubtree {i + 1}:")
        print(ast.dump(sub, indent=4))







