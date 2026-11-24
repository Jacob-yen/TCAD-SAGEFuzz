import json
import tempfile
import unittest
from pathlib import Path

from extract_spike_guards import (METADATA_DEFAULT, ExpressionParser,
                                  extract_instruction, macro_table, normalize_tree,
                                  operand_role_map)


class SpikeGuardExtractorTests(unittest.TestCase):
    def test_encoding_roles_follow_instruction_metadata_operand_order(self):
        instructions = {entry["name"]: entry for entry in json.loads(METADATA_DEFAULT.read_text())}
        vector = operand_role_map(instructions["vadd.vv"])
        self.assertEqual([vector[key]["vector_rank"] for key in ("rd", "rs2", "rs1")], [0, 1, 2])
        scalar = operand_role_map(instructions["vadd.vx"])
        self.assertEqual(scalar["rs1"]["bank"], "Integer")
        self.assertIsNone(scalar["rs1"]["vector_rank"])
        load = operand_role_map(instructions["vle32.v"])
        self.assertEqual(load["rd"]["vector_rank"], 0)
        self.assertIsNone(load["rs1"]["vector_rank"])
        indexed = operand_role_map(instructions["vluxei32.v"])
        self.assertEqual(indexed["rs2"]["vector_rank"], 1)
        compress = operand_role_map(instructions["vcompress.vm"])
        self.assertEqual([compress[key]["vector_rank"] for key in ("rd", "rs2", "rs1")], [0, 1, 2])

    def test_expression_tree_normalizes_operand_and_configuration_roles(self):
        tree = ExpressionParser("insn.rd() != insn.rs2() && P.VU.vsew <= P.VU.ELEN").parse()
        normalized = normalize_tree(tree)
        self.assertEqual(normalized["node"], "binary")
        self.assertEqual(normalized["left"]["left"], {"node": "role", "value": "vd"})
        self.assertEqual(normalized["right"]["right"], {"node": "role", "value": "ELEN"})

    def test_conditional_guard_is_recorded_but_not_executable(self):
        macros = macro_table("#define PLAIN() require_align(insn.rd(), P.VU.vflmul)\n"
                             "#define CONDITIONAL() if (flag) require_align(insn.rs2(), P.VU.vflmul)\n")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vadd_vv.h"
            path.write_text("PLAIN();\nCONDITIONAL();\n")
            guards = extract_instruction(path, macros)
        by_name = {guard["source_chain"][-1]: guard for guard in guards}
        self.assertEqual(by_name["PLAIN"]["rule"]["kind"], "lmul_alignment")
        self.assertIsNone(by_name["CONDITIONAL"]["rule"])
        self.assertFalse(by_name["CONDITIONAL"]["unconditional_context"])


if __name__ == "__main__":
    unittest.main()
