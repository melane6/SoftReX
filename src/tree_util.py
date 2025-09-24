"""
Contains utility functions for working with trees
for maniplating trees such as removing nodes or subtrees,
merging trees, etc.
"""

import ast
import random

class SplittingStrategy:
    SINGLE_STATEMENT = 1 # Removing a single statement from the tree at random at each depth level
    BLOCKS = 2 # Removing a block of statements from the tree at random at each depth level
    CONSECUTIVE_STATEMENTS = 4 # Splitting the tree into multiple parts(2), each part containing consecutive statements


class TreeType:
    AST = 1
    BST = 2


def split_program(code: str):
    tree = ast.parse(code)
    stmts = tree.body

    if len(stmts) < 1:
        return [code], []  # nothing to split

    split_index = random.randint(1, len(stmts) - 1)

    # check what we are splitting on
    print(stmts[split_index].__class__.__name__)

    # walk til find thw


    part1 = ast.Module(body=stmts[:split_index], type_ignores=[])
    part2 = ast.Module(body=stmts[split_index:], type_ignores=[])

    return part1, part2

def verify_tree(tree: ast.AST):
    """Verify that the tree is valid."""
    try:
        ast.parse(ast.unparse(tree))
    except Exception as e:
        print(f"Error: {e}")
        return False
    return True

def verify_tree_can_compile(tree: ast.AST):
    """Verify that the tree can run in a python interpreter."""
    try:
        exec(ast.unparse(tree), {})
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False
    return True


def print_tree(tree: ast.AST, code: bool = False):
    if code:
        print(ast.unparse(tree))
    else:
        print(ast.dump(tree, indent=4))




if __name__ == "__main__":
    code = """import math
def f(x):
    return math.sqrt(x)
x = f(4)
y = f(-9)
z = x + y
    """
    part1, part2 = split_program(code)
    print(verify_tree(part1))
    print_tree(part1, code=True)
    print(verify_tree_can_compile(part1))
    print(verify_tree(part2))
    print_tree(part2, code=True)
    print(verify_tree_can_compile(part2))