import ast
import unittest
from pathlib import Path

from mutant import LinesMutator
from src.load import Program, LinesProgram, ASTProgram


class Mutation(unittest.TestCase):
    def setUp(self):
        self.program_path = Path("/home/melanenavaratnarajah/Documents/SoftReX/tests/data/equl.py")
        self.test_path = Path("/home/melanenavaratnarajah/Documents/SoftReX/tests/data/test_equl.py")

    def test_mutation_create(self):
        program = Program(self.program_path)
        program.test_dataset(str(self.test_path))
        lines_program = LinesProgram(path=self.program_path)
        lines_program.test_dataset(str(self.test_path))
        print(lines_program.lines)
        firstLine = lines_program.lines[0]
        mutator = LinesMutator(lines_program)
        mutation = mutator.mutate_range(0, 1, "")
        mutator.apply_mutation(mutation)
        print(lines_program.lines)
        assert lines_program.lines[0] != firstLine


