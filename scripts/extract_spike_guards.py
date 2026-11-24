#!/usr/bin/env python3
"""Extract RVV guard-call expression trees from Spike instruction/macro sources.

This parser covers the explicit guard-call subset used by the RVV rule catalog.
Unparsed expressions remain in the output with their source text; they are not
silently converted into a legality rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


TOKEN = re.compile(
    r"\s*(->|&&|\|\||==|!=|<=|>=|[A-Za-z_]\w*|(?:\d+\.\d+|\d+)|[.()+*/%!,<>\-])"
)
IDENT = re.compile(r"[A-Za-z_]\w*")
DECL = re.compile(r"(?:const\s+)?(?:reg_t|float|double|int|size_t|uint\d+_t)\s+(\w+)\s*=\s*([^;]+);")
PRECEDENCE = {"||": 1, "&&": 2, "==": 3, "!=": 3, "<": 3,
              "<=": 3, ">": 3, ">=": 3, "+": 4, "-": 4,
              "*": 5, "/": 5, "%": 5}
GUARDS = ("require", "require_align", "require_vector", "require_vector_novtype")
ROLES = {"insn.rd": "vd", "insn.rs1": "vs1", "insn.rs2": "vs2",
         "insn.v_nf": "NF_minus_one", "P.VU.vsew": "SEW",
         "P.VU.vflmul": "LMUL", "P.VU.ELEN": "ELEN",
         "P.VU.vlenb": "VLENB", "P.VU.vl": "VL",
         "P.VU.vstart": "VSTART"}
OPERAND_ROLES = {"vd", "vs1", "vs2", "NF_minus_one"}
CONFIG_ROLES = {"SEW", "LMUL", "ELEN", "VLENB", "VL", "VSTART"}
METADATA_DEFAULT = (Path(__file__).resolve().parents[1]
                    / "third_party/riscv-instruction-crates/assets/riscv_instructions.json")


def operand_role_map(instruction: dict) -> dict[str, dict]:
    operands = {operand["name"]: operand for operand in instruction.get("operands", [])}
    syntax = instruction.get("assembly_syntax", {})
    if "Format" in syntax:
        assembly = syntax["Format"]
    else:
        snippets = re.findall(r'format!\(\s*"([^"]+)"', syntax.get("RustCode", ""))
        assembly = next((snippet for snippet in snippets
                         if snippet.startswith(instruction["name"] + " ")), "")
    ordered = [token.removesuffix("_enc") for token in re.findall(r"\{([A-Za-z_]\w*)\}", assembly)]
    vector_names = [name for name in ordered
                    if name in operands and operands[name].get("kind", {}).get("Register", {}).get("bank") == "Vector"]
    result = {}
    for encoding, choices in (("rd", ("vd", "vs3")), ("rs2", ("vs2", "xs2", "fs2")),
                              ("rs1", ("vs1", "xs1", "fs1"))):
        name = next((name for name in choices if name in operands), None)
        if name is None:
            continue
        bank = operands[name].get("kind", {}).get("Register", {}).get("bank")
        result[encoding] = {"name": name, "bank": bank,
                            "vector_rank": vector_names.index(name) if name in vector_names else None}
    return result


def strip_comments(source: str) -> str:
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    return re.sub(r"//[^\n]*", "", source)


def logical_lines(source: str) -> list[str]:
    lines, pending = [], ""
    for line in source.splitlines():
        stripped = line.rstrip()
        if stripped.endswith("\\"):
            pending += stripped[:-1] + " "
        else:
            lines.append(pending + stripped)
            pending = ""
    if pending:
        lines.append(pending)
    return lines


def split_args(value: str) -> list[str]:
    args, depth, start = [], 0, 0
    for i, char in enumerate(value):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            args.append(value[start:i].strip())
            start = i + 1
    tail = value[start:].strip()
    if tail:
        args.append(tail)
    return args


def calls(source: str, names: set[str]) -> list[tuple[str, list[str], int]]:
    found = []
    for match in re.finditer(r"\b([A-Za-z_]\w*)\s*\(", source):
        name = match.group(1)
        if name not in names:
            continue
        depth, end = 1, match.end()
        while end < len(source) and depth:
            if source[end] == "(":
                depth += 1
            elif source[end] == ")":
                depth -= 1
            end += 1
        if depth == 0:
            found.append((name, split_args(source[match.end():end - 1]), match.start()))
    return found


class ExpressionParser:
    def __init__(self, expression: str):
        self.tokens = []
        cursor = 0
        while cursor < len(expression):
            match = TOKEN.match(expression, cursor)
            if not match:
                if expression[cursor:].strip():
                    raise ValueError(f"unsupported token: {expression[cursor:cursor + 24]}")
                break
            self.tokens.append(match.group(1))
            cursor = match.end()
        self.at = 0

    def peek(self) -> str | None:
        return self.tokens[self.at] if self.at < len(self.tokens) else None

    def pop(self) -> str:
        value = self.peek()
        if value is None:
            raise ValueError("incomplete expression")
        self.at += 1
        return value

    def expression(self, minimum: int = 0) -> dict:
        token = self.pop()
        if token in ("!", "-"):
            left = {"node": "unary", "op": token, "arg": self.expression(6)}
        elif token == "(":
            left = self.expression()
            if self.pop() != ")":
                raise ValueError("unmatched parenthesis")
        elif re.fullmatch(r"\d+(?:\.\d+)?", token):
            left = {"node": "number", "value": token}
        elif IDENT.fullmatch(token):
            parts = [token]
            while self.peek() in (".", "->"):
                self.pop()
                parts.append(self.pop())
            name = ".".join(parts)
            if self.peek() == "(":
                self.pop()
                args = []
                while self.peek() != ")":
                    args.append(self.expression())
                    if self.peek() != ",":
                        break
                    self.pop()
                if self.pop() != ")":
                    raise ValueError("invalid call")
                left = {"node": "call", "name": name, "args": args}
            else:
                left = {"node": "name", "value": name}
        else:
            raise ValueError(f"unexpected token: {token}")
        while self.peek() in PRECEDENCE and PRECEDENCE[self.peek()] >= minimum:
            op = self.pop()
            right = self.expression(PRECEDENCE[op] + 1)
            left = {"node": "binary", "op": op, "left": left, "right": right}
        return left

    def parse(self) -> dict:
        tree = self.expression()
        if self.peek() is not None:
            raise ValueError(f"unparsed token: {self.peek()}")
        return tree


def expand_aliases(expression: str, aliases: dict[str, str]) -> str:
    for _ in range(6):
        replaced = expression
        for name, value in aliases.items():
            replaced = re.sub(rf"\b{re.escape(name)}\b", f"({value})", replaced)
        if replaced == expression:
            break
        expression = replaced
    return expression


def macro_table(header: str) -> dict[str, tuple[list[str], str]]:
    table = {}
    for line in logical_lines(strip_comments(header)):
        match = re.match(r"\s*#define\s+(\w+)(?:\(([^)]*)\))?\s+(.*)", line)
        if match:
            table[match.group(1)] = (split_args(match.group(2) or ""), match.group(3))
    return table


def normalized_rule(mnemonic: str, guard: str, arguments: list[str], unconditional: bool) -> dict | None:
    if guard == "require" and len(arguments) == 1:
        compact = re.sub(r"\s+", "", arguments[0])
        if compact == "insn.rd()!=insn.rs2()":
            return {"kind": "register_base_inequality", "left": "vd", "right": "vs2"}
    if guard != "require_align" or len(arguments) != 2:
        return None
    match = re.match(r"(?:vl|vs)([1248])(?:re\d+|r)\.v$", mnemonic)
    if match and "insn.rd()" in arguments[0]:
        return {"kind": "group_alignment", "operand": "vd_or_vs3",
                "group_size": int(match.group(1))}
    if unconditional:
        operand = re.search(r"insn\.(rd|rs1|rs2)\(\)", arguments[0])
        width = re.sub(r"[\s()]", "", arguments[1])
        scale = re.fullmatch(r"P\.VU\.vflmul(?:([*/])(\d+))?", width)
        if operand and scale:
            operator, factor = scale.group(1), int(scale.group(2) or 1)
            return {"kind": "lmul_alignment", "operand": operand.group(1),
                    "scale_num": factor if operator == "*" else 1,
                    "scale_den": factor if operator == "/" else 1}
    return None


def normalize_tree(tree: dict) -> dict:
    kind = tree["node"]
    if kind in ("name", "call"):
        value = tree.get("value", tree.get("name"))
        if value in ROLES and (kind == "name" or not tree.get("args")):
            return {"node": "role", "value": ROLES[value]}
    if kind == "binary":
        return {**tree, "left": normalize_tree(tree["left"]),
                "right": normalize_tree(tree["right"])}
    if kind == "unary":
        return {**tree, "arg": normalize_tree(tree["arg"])}
    if kind == "call":
        return {**tree, "args": [normalize_tree(arg) for arg in tree["args"]]}
    return tree


def roles_in(tree: dict) -> set[str]:
    if tree["node"] == "role":
        return {tree["value"]}
    if tree["node"] == "binary":
        return roles_in(tree["left"]) | roles_in(tree["right"])
    if tree["node"] == "unary":
        return roles_in(tree["arg"])
    if tree["node"] == "call":
        return set().union(*(roles_in(arg) for arg in tree["args"]))
    return set()


def category(guard: str, trees: list[dict]) -> str:
    if any(tree["node"] == "unparsed" for tree in trees):
        return "unparsed"
    roles = set().union(*(roles_in(tree) for tree in trees))
    operand, config = bool(roles & OPERAND_ROLES), bool(roles & CONFIG_ROLES)
    if operand and config:
        return "joint"
    if operand or guard == "require_align":
        return "C_operand"
    if config or guard in ("require_vector", "require_vector_novtype"):
        return "C_config"
    return "other"


def extract_instruction(path: Path, table: dict[str, tuple[list[str], str]],
                        metadata: dict | None = None) -> list[dict]:
    mnemonic = path.stem.replace("_", ".")
    source = strip_comments(path.read_text(errors="replace"))
    guards = []
    seen = set()
    roles = operand_role_map(metadata) if metadata else {}

    def visit(body: str, chain: list[str], depth: int, guarded: bool = False) -> None:
        if depth > 12:
            return
        aliases = dict(DECL.findall(body))
        for name, arguments, offset in calls(body, set(GUARDS)):
            expanded = [expand_aliases(value, aliases) for value in arguments]
            key = (name, tuple(expanded), tuple(chain))
            if key in seen:
                continue
            seen.add(key)
            trees = []
            for expression in expanded:
                try:
                    trees.append(ExpressionParser(expression).parse())
                except ValueError as error:
                    trees.append({"node": "unparsed", "source": expression, "reason": str(error)})
            normalized = [normalize_tree(tree) for tree in trees]
            unconditional = not guarded and not re.search(r"\bif\b|\?", body[:offset])
            rule = normalized_rule(mnemonic, name, expanded, unconditional)
            if rule and rule["kind"] in ("lmul_alignment", "group_alignment"):
                encoding = "rd" if rule["kind"] == "group_alignment" else rule["operand"]
                rule["vector_rank"] = roles.get(encoding, {}).get("vector_rank")
            if rule and rule["kind"] == "register_base_inequality":
                rule["left_vector_rank"] = roles.get("rd", {}).get("vector_rank")
                rule["right_vector_rank"] = roles.get("rs2", {}).get("vector_rank")
            guards.append({"guard": name, "arguments": expanded, "ast": trees,
                           "normalized_ast": normalized,
                           "category": category(name, normalized),
                           "source_chain": chain,
                           "unconditional_context": unconditional,
                           "rule": rule})
        for name, arguments, offset in calls(body, set(table)):
            if name in chain:
                continue
            parameters, definition = table[name]
            if len(parameters) != len(arguments):
                continue
            expanded = definition
            for parameter, argument in zip(parameters, arguments):
                expanded = re.sub(rf"\b{re.escape(parameter)}\b", f"({argument})", expanded)
            child_guarded = guarded or bool(re.search(r"\bif\b|\?", body[:offset]))
            visit(expanded, [*chain, name], depth + 1, child_guarded)
        for name in set(IDENT.findall(body)) & set(table):
            parameters, definition = table[name]
            if not parameters and name not in chain:
                child_guarded = guarded or bool(re.search(r"\bif\b|\?", body[:body.find(name)]))
                visit(definition, [*chain, name], depth + 1, child_guarded)

    visit(source, [path.name], 0)
    return guards


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spike-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=METADATA_DEFAULT)
    args = parser.parse_args()
    source = args.spike_source.resolve()
    header = source / "riscv/v_ext_macros.h"
    instructions = source / "riscv/insns"
    if not header.is_file() or not instructions.is_dir():
        parser.error("expected riscv/v_ext_macros.h and riscv/insns under Spike source")
    table = macro_table(header.read_text(errors="replace"))
    metadata_bytes = args.metadata.read_bytes()
    metadata = {item["name"]: item for item in json.loads(metadata_bytes)}
    entries = {}
    for path in sorted(instructions.glob("v*.h")):
        guards = extract_instruction(path, table, metadata.get(path.stem.replace("_", ".")))
        if guards:
            entries[path.stem.replace("_", ".")] = guards
    commit = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=False).stdout.strip()
    result = {"spike_commit": commit or None,
              "instruction_metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
              "instruction_files": len(entries),
              "macros_examined": len(table), "instructions": entries}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(entries)} instruction guard sets to {args.out}")


if __name__ == "__main__":
    main()
