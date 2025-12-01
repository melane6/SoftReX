import ast
import unittest
from pathlib import Path
from src.load import Program, LinesProgram, ASTProgram


class Load(unittest.TestCase):
    def test_load_file(self):
        program = Program(Path("/home/melanenavaratnarajah/Documents/SoftReX/tests/data/test_equl.py"))
        assert program.path == Path("/home/melanenavaratnarajah/Documents/SoftReX/tests/data/test_equl.py")
        assert program.source is not None

    def test_load_source(self):
        source_code = "def add(a, b):\n    return a + b\n"
        program = Program()
        program.from_source(source_code)
        assert program.source == source_code


    def test_dump(self):
        source_code = "def add(a, b):\n    return a + b\n"
        program = Program()
        program.from_source(source_code)
        dumped_code = program.dump()
        assert dumped_code == source_code

    def test_lines_program(self):
        lines = ["def add(a, b):", "    return a + b"]
        source_code = "\n".join(lines)
        program = LinesProgram(source=source_code)
        lines_program = program
        assert lines_program.source == source_code
        assert lines_program.path is None
        assert lines_program.lines == lines
        subsets = lines_program.subsets
        assert subsets == []
        subsets = lines_program.generate_subsets(subset_size=2)
        assert len(subsets[0]) == 2
        subsets = lines_program.generate_subsets(subset_size=3)
        assert len(subsets[0]) == 2 # only 2 lines available

    def test_ast_program(self):
        source_code = "def add(a, b):\n    return a + b\n"
        tree = ast.parse(source_code)
        nodes = ast.walk(tree)
        program = ASTProgram(source=source_code)
        assert program.source == source_code
        assert program.path is None
        assert ast.dump(program.tree) == ast.dump(tree)
        print(program.collect_nodes())
        print(list(nodes))
        assert len(program.collect_nodes()) == len(list(nodes))

        for i, node in enumerate(program.collect_nodes()):
            assert isinstance(node, ast.AST)
            print(f"Node {i}: {node.__class__.__name__}, AST Walk: {list(ast.walk(tree))[i].__class__.__name__}")
            assert node.__class__.__name__ == list(ast.walk(tree))[i].__class__.__name__



if __name__ == '__main__':
    unittest.main()