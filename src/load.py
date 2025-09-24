import os
import logging
import ast
import random
from enum import Enum

class Language(Enum):
    PYTHON = 1
    PYTHON_TEST_SUITE = 2
    JAVA = 3

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







