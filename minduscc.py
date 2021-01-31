#!/usr/bin/env python3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from secrets import token_hex
from typing import List, Any

from sly import Lexer, Parser


@contextmanager
def next_token(state):
    token = "r" + token_hex(4)
    state['tokens'].append(token)
    try:
        yield token
    finally:
        state['tokens'].pop()


def indent(s, amount=1):
    return str(s).replace('\n', '\n' + '  '*amount)


def main(sources, output):
    lexer = MindusLexer()
    parser = MindusParser()
    asm = []
    index = 0

    for source in sources:
        with open(source, "r") as f:
            raw = f.read()

        # Lex and Parse raw source into AST
        ast = parser.parse(lexer.tokenize(raw))
        if not ast:
            print("error: compilation failed on", source)
            return

        # Compile into instructions
        instructions = ast.compile()

        # Optimize up to 16 times
        for i in range(16):
            if optimize(instructions) == 0:
                break

        # Emit asm
        for instruction in instructions:
            instruction.index = index
            asm.append(instruction.emit())
            index += 1

    # Output script
    script = "\n".join(asm) + "\n"
    with open(output, "w") as f:
        print(script, file=f)


def optimize(instructions):
    optimizations = 0
    i = 0
    prev_instruction = None
    while i < len(instructions):
        # Get references to next and current instruction
        if i + 1 < len(instructions):
            next_instruction = instructions[i+1]
        else:
            next_instruction = None
        instruction = instructions[i]
        # Check for sensor directly into set
        if (isinstance(instruction, SensorInstruction) and
                isinstance(next_instruction, SetInstruction) and
                instruction.output == next_instruction.val):
            instruction.output = next_instruction.output
            instructions.pop(i + 1)
            optimizations += 1
        # Check for set directly into op
        if (isinstance(instruction, SetInstruction) and
                isinstance(next_instruction, OpInstruction) and
                (instruction.output == next_instruction.right or
                instruction.output == next_instruction.left)):
            if instruction.output == next_instruction.right:
                next_instruction.right = instruction.val
            else:
                next_instruction.left = instruction.val
            instructions.pop(i)
            optimizations += 1
        # Check for jumpable op directly into jump
        if (
                isinstance(instruction, OpInstruction)
                and
                isinstance(next_instruction, JumpInstruction)
                and
                instruction.op in jump_operations
                and
                next_instruction.op == 'greaterThan'
                and
                instruction.output == next_instruction.left
                and
                next_instruction.right == 0
                    ):
            next_instruction.left = instruction.left
            next_instruction.right = instruction.right
            next_instruction.op = instruction.op
            instructions.pop(i)
            optimizations += 1
        # Advance to the next instruction
        else:
            i += 1
            prev_instruction = instruction
    return optimizations


class MindusLexer(Lexer):
    # Set of tokens
    tokens = {
        IF, ELSE, ELIF,
        IDENTIFIER, NUMBER_LITERAL, STRING_LITERAL,
        ADD, SUBTRACT, MULTIPLY, DIVIDE, INT_DIVIDE, MODULUS,
        GREATER_THAN, LESS_THAN, GREATER_THAN_EQ, LESS_THAN_EQ,
        EQUALS, NOT_EQUALS, LOGICAL_AND, LOGICAL_OR,
        BITWISE_AND, BITWISE_XOR, BITWISE_OR,
        ASSIGN,
        OPEN_PAREN, CLOSE_PAREN,
        OPEN_BRACE, CLOSE_BRACE,
        SEMICOLON, COMMA
    }

    # Characters ignored between tokens
    ignore = ' \t'
    ignore_comment = r'\#.*'

    @_(r'\n+')
    def ignore_newline(self, t):
        self.lineno += len(t.value)

    # Regex Rules for tokens
    IDENTIFIER      = r'[a-zA-Z_.][a-zA-Z0-9_.]*'

    @_(r'\d+|0x[0-9a-fA-F]+')
    def NUMBER_LITERAL(self, t):
        if t.value[:2] == '0x':
            t.value = int(t.value, 16)
        else:
            t.value = int(t.value)
        return t

    @_(r'"[^"]*"' + '|' + r"'[^']*'")
    def STRING_LITERAL(self, t):
        t.value = t.value[1:-1]
        return t

    ADD             = r'\+'
    SUBTRACT        = r'-'
    MULTIPLY        = r'\*'
    DIVIDE          = r'/'
    INT_DIVIDE      = r'//'
    MODULUS         = r'%'
    GREATER_THAN_EQ = r'>='
    GREATER_THAN    = r'>'
    LESS_THAN_EQ    = r'<='
    LESS_THAN       = r'<'
    EQUALS          = r'=='
    NOT_EQUALS      = r'!='
    BITWISE_AND     = r'&'
    BITWISE_XOR     = r'\^'
    BITWISE_OR      = r'\|'
    LOGICAL_AND     = r'&&'
    LOGICAL_OR      = r'\|\|'
    ASSIGN          = r'='
    OPEN_PAREN      = r'\('
    CLOSE_PAREN     = r'\)'
    OPEN_BRACE      = r'\{'
    CLOSE_BRACE     = r'\}'
    SEMICOLON       = r';'
    COMMA           = r','

    IDENTIFIER['if'] = IF
    IDENTIFIER['else'] = ELSE
    IDENTIFIER['elif'] = ELIF


class MindusParser(Parser):
    start = 'program'
    tokens = MindusLexer.tokens
    precedence = (
        ('right', ASSIGN),
        ('left', LOGICAL_OR),
        ('left', LOGICAL_AND),
        ('left', BITWISE_OR),
        ('left', BITWISE_XOR),
        ('left', BITWISE_AND),
        ('nonassoc', EQUALS, NOT_EQUALS),
        ('nonassoc', GREATER_THAN, LESS_THAN, GREATER_THAN_EQ, LESS_THAN_EQ),
        ('left', ADD, SUBTRACT),
        ('left', MULTIPLY, DIVIDE, INT_DIVIDE, MODULUS),
    )

    @_('expr MODULUS expr')
    def expr(self, p):
        if isinstance(p[0], NumberNode) and isinstance(p[2], NumberNode):
            return NumberNode(p[0].value % p[2].value)
        return ModulusOperation(p[0], p[2])

    @_('expr INT_DIVIDE expr')
    def expr(self, p):
        if isinstance(p[0], NumberNode) and isinstance(p[2], NumberNode):
            return NumberNode(p[0].value // p[2].value)
        return IntDivideOperation(p[0], p[2])

    @_('expr LOGICAL_OR expr')
    def expr(self, p):
        if isinstance(p[0], (StringNode, NumberNode)) and isinstance(p[2], (StringNode, NumberNode)):
            return NumberNode(int(p[0].value or p[2].value))
        return LogicalOrOperation(p[0], p[2])

    @_('expr LOGICAL_AND expr')
    def expr(self, p):
        if isinstance(p[0], (StringNode, NumberNode)) and isinstance(p[2], (StringNode, NumberNode)):
            return NumberNode(int(p[0].value and p[2].value))
        return LogicalAndOperation(p[0], p[2])

    @_('expr BITWISE_OR expr')
    def expr(self, p):
        if isinstance(p[0], NumberNode) and isinstance(p[2], NumberNode):
            return NumberNode(p[0].value | p[2].value)
        return BitwiseOrOperation(p[0], p[2])

    @_('expr BITWISE_XOR expr')
    def expr(self, p):
        if isinstance(p[0], NumberNode) and isinstance(p[2], NumberNode):
            return NumberNode(p[0].value ^ p[2].value)
        return BitwiseXorOperation(p[0], p[2])

    @_('expr BITWISE_AND expr')
    def expr(self, p):
        if isinstance(p[0], NumberNode) and isinstance(p[2], NumberNode):
            return NumberNode(p[0].value & p[2].value)
        return BitwiseAndOperation(p[0], p[2])

    @_('expr ADD expr')
    def expr(self, p):
        if isinstance(p[0], NumberNode) and isinstance(p[2], NumberNode):
            return NumberNode(p[0].value + p[2].value)
        elif isinstance(p[0], StringNode) and isinstance(p[2], StringNode):
            return StringNode(p[0].value + p[2].value)
        return AddOperation(p[0], p[2])

    @_('expr SUBTRACT expr')
    def expr(self, p):
        if isinstance(p[0], NumberNode) and isinstance(p[2], NumberNode):
            return NumberNode(p[0].value - p[2].value)
        return SubtractOperation(p[0], p[2])

    @_('expr MULTIPLY expr')
    def expr(self, p):
        if isinstance(p[0], NumberNode) and isinstance(p[2], NumberNode):
            return NumberNode(p[0].value * p[2].value)
        elif isinstance(p[0], StringNode) and isinstance(p[2], NumberNode):
            return StringNode(p[0].value * p[2].value)
        elif isinstance(p[0], NumberNode) and isinstance(p[2], StringNode):
            return StringNode(p[2].value * p[0].value)
        return MultiplyOperation(p[0], p[2])

    @_('expr DIVIDE expr')
    def expr(self, p):
        if isinstance(p[0], NumberNode) and isinstance(p[2], NumberNode):
            return NumberNode(p[0].value / p[2].value)
        return DivideOperation(p[0], p[2])

    @_('expr EQUALS expr')
    def expr(self, p):
        if isinstance(p[0], NumberNode) and isinstance(p[2], NumberNode):
            return NumberNode(int(p[0].value == p[2].value))
        elif isinstance(p[0], StringNode) and isinstance(p[2], StringNode):
            return NumberNode(int(p[0].value == p[2].value))
        return EqualCompare(p[0], p[2])

    @_('expr NOT_EQUALS expr')
    def expr(self, p):
        if isinstance(p[0], NumberNode) and isinstance(p[2], NumberNode):
            return NumberNode(int(p[0].value != p[2].value))
        elif isinstance(p[0], StringNode) and isinstance(p[2], StringNode):
            return NumberNode(int(p[0].value != p[2].value))
        return NotEqualCompare(p[0], p[2])

    @_('expr GREATER_THAN expr')
    def expr(self, p):
        if isinstance(p[0], NumberNode) and isinstance(p[2], NumberNode):
            return NumberNode(int(p[0].value > p[2].value))
        return GreaterThanCompare(p[0], p[2])

    @_('expr GREATER_THAN_EQ expr')
    def expr(self, p):
        if isinstance(p[0], NumberNode) and isinstance(p[2], NumberNode):
            return NumberNode(int(p[0].value >= p[2].value))
        return GreaterThanEqCompare(p[0], p[2])

    @_('expr LESS_THAN expr')
    def expr(self, p):
        if isinstance(p[0], NumberNode) and isinstance(p[2], NumberNode):
            return NumberNode(int(p[0].value < p[2].value))
        return LessThanCompare(p[0], p[2])

    @_('expr LESS_THAN_EQ expr')
    def expr(self, p):
        if isinstance(p[0], NumberNode) and isinstance(p[2], NumberNode):
            return NumberNode(int(p[0].value <= p[2].value))
        return LessThanEqCompare(p[0], p[2])

    @_('OPEN_PAREN expr CLOSE_PAREN')
    def expr(self, p):
        return p[1]

    @_('NUMBER_LITERAL')
    def expr(self, p):
        return NumberNode(p[0])

    @_('STRING_LITERAL')
    def expr(self, p):
        return StringNode(p[0])

    @_('IDENTIFIER')
    def expr(self, p):
        return Variable(p[0])

    @_('function_call')
    def expr(self, p):
        return p[0]

    @_('')
    def arguments(self, p):
        return []

    @_('expr')
    def arguments(self, p):
        return [p[0]]

    @_('arguments COMMA expr')
    def arguments(self, p):
        p[0].append(p[2])
        return p[0]

    @_('IDENTIFIER OPEN_PAREN arguments CLOSE_PAREN')
    def function_call(self, p):
        return FunctionCall(p[0], p[2])

    @_('IDENTIFIER ASSIGN expr')
    def assignment(self, p):
        return Assignment(p[0], p[2])

    @_('assignment SEMICOLON', 'function_call SEMICOLON')
    def statement(self, p):
        return p[0]

    @_('statement', 'if_statement')
    def block(self, p):
        return [p[0]]

    @_('block if_statement', 'block statement')
    def block(self, p):
        p[0].append(p[1])
        return p[0]

    @_('IF OPEN_PAREN expr CLOSE_PAREN OPEN_BRACE block CLOSE_BRACE')
    def if_statement(self, p):
        return Conditional(p.expr, p.block, None)

    @_('if_statement elif_statement')
    def if_statement(self, p):
        node = p[0]
        while node.false_block:
            node = node.false_block
        node.false_block = p[1]
        return p[0]

    @_('if_statement else_statement')
    def if_statement(self, p):
        node = p[0]
        while node.false_block:
            node = node.false_block
        node.false_block = p[1]
        return p[0]

    @_('ELIF OPEN_PAREN expr CLOSE_PAREN OPEN_BRACE block CLOSE_BRACE')
    def elif_statement(self, p):
        return Conditional(p.expr, p.block, None)

    @_('ELSE OPEN_BRACE block CLOSE_BRACE')
    def else_statement(self, p):
        return p.block

    @_('statement', 'if_statement')
    def program(self, p):
        return Program([p[0]])

    @_('program if_statement', 'program statement')
    def program(self, p):
        p.program.units.append(p[1])
        return p.program


def compile_block(block, state):
    instructions = []
    for statement in block:
        instructions.extend(statement.compile(state))
    return instructions


@dataclass
class Instruction:
    index: int = None


class SetInstruction(Instruction):
    def __init__(self, output, val):
        super().__init__()
        self.output = output
        self.val = val

    def emit(self):
        return "set {} {}".format(
            self.output,
            self.val
        )


class OpInstruction(Instruction):
    def __init__(self, op, output, left, right):
        super().__init__()
        self.op = op
        self.output = output
        self.left = left
        self.right = right

    def emit(self):
        return "op {} {} {} {}".format(
            self.op,
            self.output,
            self.left,
            self.right
        )

jump_operations = {
    "equal", "notEqual",
    "lessThan", "lessThanEq",
    "greaterThan", "greaterThanEq",
}
class JumpInstruction(Instruction):
    def __init__(self, offset, op, left, right):
        super().__init__()
        self.offset = offset
        self.op = op
        self.left = left
        self.right = right

    def emit(self):
        return "jump {} {} {} {}".format(
            self.index + self.offset,
            self.op,
            self.left,
            self.right
        )


class GotoInstruction(Instruction):
    def __init__(self, offset):
        super().__init__()
        self.offset = offset

    def emit(self):
        return "jump {} always 0 0".format(self.index + self.offset)


class ReturnInstruction(Instruction):
    def emit(self):
        return "end"


class CallInstruction(Instruction):
    def __init__(self, name, *arguments):
        super().__init__()
        self.name = name
        self.arguments = arguments

    def emit(self):
        return "{} {}".format(
            self.name,
            " ".join(str(arg) for arg in self.arguments)
        )


class SensorInstruction(Instruction):
    def __init__(self, output, structure, resource):
        super().__init__()
        self.output = output
        self.structure = structure
        self.resource = resource

    def emit(self):
        return "sensor {} {} {}".format(
            self.output,
            self.structure,
            self.resource
        )


@dataclass
class Program:
    units: List[Any]
    def __str__(self):
        return "Program(\n  units=[\n    {}\n  ]\n)".format(
            indent(",\n  ".join(indent(s) for s in self.units))
        )

    def compile(self, state=None):
        if state is None:
            state = {"tokens": ['discard']}
        instructions = compile_block(self.units, state)
        instructions.append(ReturnInstruction())
        return instructions


@dataclass
class FunctionCall:
    name: str
    arguments: List[Any]

    def print_flush_compile(self, state):
        if len(self.arguments) != 1:
            raise ValueError("print_flush() requires one string argument")
        if not isinstance(self.arguments[0], StringNode):
            raise ValueError("print_flush() requires one string argument")

        return [CallInstruction(
            'printflush',
            self.arguments[0].value
        )]

    def print_compile(self, state):
        instructions = []
        for arg in self.arguments:
            if isinstance(arg, StringNode):
                instructions.append(CallInstruction(
                    'print',
                    '"{}"'.format(arg.value)
                ))
            elif isinstance(arg, Variable):
                instructions.append(CallInstruction(
                    'print',
                    arg.name
                ))
            else:
                with next_token(state) as token:
                    instructions.extend(arg.compile(state))
                    instructions.append(CallInstruction(
                        'print',
                        token
                    ))
        return instructions

    def control_enabled_compile(self, state, status):
        instructions = []
        for arg in self.arguments:
            if not isinstance(arg, StringNode):
                raise TypeError("cannot call enable/disable on non-string literal")
            instructions.append(CallInstruction(
                'control',
                'enabled {} {} 0 0 0'.format(arg.value, status)
            ))
        return instructions

    def sensor_compile(self, state):
        if len(self.arguments) != 2:
            raise ValueError("sensor() requires two arguments")

        for arg in self.arguments:
            if not isinstance(arg, StringNode):
                raise TypeError("sensor() requires two string literal argumentss")

        token = state['tokens'][-1]
        return [SensorInstruction(
            token,
            self.arguments[0].value,
            "@{}".format(self.arguments[1].value)
        )]

    def compile(self, state):
        if self.name == 'print':
            return self.print_compile(state)
        elif self.name == 'sensor':
            return self.sensor_compile(state)
        elif self.name == 'enable':
            return self.control_enabled_compile(state, 1)
        elif self.name == 'disable':
            return self.control_enabled_compile(state, 0)
        elif self.name == 'print_flush':
            return self.print_flush_compile(state)
        else:
            return [CallInstruction(
                self.name,
                *self.arguments
            )]


@dataclass
class Assignment:
    identifier: str
    value: Any

    def compile(self, state):
        with next_token(state) as token:
            instructions = self.value.compile(state)
            instructions.append(SetInstruction(self.identifier, token))
        return instructions


@dataclass
class Conditional:
    condition: Any
    true_block: Any
    false_block: Any

    def __str__(self):
        return "Cond(\n  expr={}\n  true={}\n  false={}\n)".format(
            indent(self.condition),
            indent(self.true_block),
            indent(self.false_block)
        )

    def compile(self, state):
        true_instructions = compile_block(self.true_block, state)
        if isinstance(self.false_block, Conditional):
            false_instructions = self.false_block.compile(state)
        else:
            false_instructions = compile_block(self.false_block, state)

        with next_token(state) as token:
            condition_instructions = self.condition.compile(state)
            jump_instruction = JumpInstruction(
                len(false_instructions) + 2, # Offset
                "greaterThan", # Operation
                token, # Left
                0 # Right
            )

        instructions = []
        instructions.extend(condition_instructions)
        instructions.append(jump_instruction)
        instructions.extend(false_instructions)
        instructions.append(GotoInstruction(len(true_instructions) + 1))
        instructions.extend(true_instructions)
        return instructions


@dataclass
class NumberNode:
    value: int

    def compile(self, state):
        token = state['tokens'][-1]
        return [SetInstruction(token, self.value)]


@dataclass
class StringNode:
    value: str


@dataclass
class Variable:
    name: str

    def compile(self, state):
        token = state['tokens'][-1]
        return [SetInstruction(token, self.name)]


@dataclass
class BinaryOperation:
    left: Any
    right: Any

    def compile(self, state):
        token = state['tokens'][-1]
        instructions = []

        with next_token(state) as left_token:
            instructions.extend(self.left.compile(state))

            with next_token(state) as right_token:
                instructions.extend(self.right.compile(state))

                instructions.append(
                    OpInstruction(
                        self.operation, token, left_token, right_token
                    )
                )

        return instructions


class AddOperation(BinaryOperation):
    operation = 'add'


class SubtractOperation(BinaryOperation):
    operation = 'sub'


class MultiplyOperation(BinaryOperation):
    operation = 'mul'


class ModulusOperation(BinaryOperation):
    operation = 'mod'


class DivideOperation(BinaryOperation):
    operation = 'div'


class IntDivideOperation(BinaryOperation):
    operation = 'idiv'


class LogicalOrOperation(BinaryOperation):
    operation = 'or'


class LogicalAndOperation(BinaryOperation):
    operation = 'land'


class BitwiseOrOperation(BinaryOperation):
    operation = 'or'


class BitwiseXorOperation(BinaryOperation):
    operation = 'xor'


class BitwiseAndOperation(BinaryOperation):
    operation = 'and'


class GreaterThanCompare(BinaryOperation):
    operation = 'greaterThan'


class GreaterThanEqCompare(BinaryOperation):
    operation = 'greaterThanEq'


class LessThanCompare(BinaryOperation):
    operation = 'lessThan'


class LessThanEqCompare(BinaryOperation):
    operation = 'lessThanEq'


class EqualCompare(BinaryOperation):
    operation = 'equal'


class NotEqualCompare(BinaryOperation):
    operation = 'notEqual'


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('source', nargs='+', type=Path)
    parser.add_argument('-o', '--output', default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = args.source[0].with_suffix('.mlog')

    main(args.source, args.output)
