"""
Generate a responsibility map based on the tree like structure of the code.
"""
import ast


class ResponsibilityMap:
    def __init__(self, tree):
        self.tree = tree
        self.map = {}
        self.traces = {}

    def generate(self):
        self._traverse(self.tree)
        return self.map

    def _traverse(self, node, parent=None):
        node_type = type(node).__name__
        if node_type not in self.map:
            self.map[node_type] = []
        if parent:
            self.map[node_type].append(type(parent).__name__)
        for child in ast.iter_child_nodes(node):
            self._traverse(child, node)