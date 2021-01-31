#!/usr/bin/env python3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from prettyprinter import pprint, install_extras
from secrets import token_hex
from typing import List, Any

from sly import Lexer, Parser


# Pretty printer dataclasses support
install_extras(include=['dataclasses'], warn_on_error=True)


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


def main(sources, output, verbose):
    lexer = MindusLexer()
    parser = MindusParser()
    asm = []
    index = 0

    for source in sources:
        with open(source, "r") as f:
            raw = f.read()

        # Lex and parse raw source into AST
        ast = parser.parse(lexer.tokenize(raw))
        if not ast:
            print("error: compilation failed on", source)
            return

        if verbose:
            pprint(ast)

        # Compile into instructions
        instructions = ast.compile()

        # Store jump targets and jump sources for each instruction
        for instruction in instructions:
            instruction.jump_sources = set()
        for i, instruction in enumerate(instructions):
            if isinstance(instruction, JumpInstruction):
                instruction.jump_target = instructions[i + instruction.offset]
                instructions[i + instruction.offset].jump_sources.add(instruction)

        # Assign canonical indices
        for instruction in instructions:
            instruction.index = index
            index += 1

        # Emit asm
        for instruction in instructions:
            asm.append(instruction.emit())

    # Output script
    script = "\n".join(asm)
    with open(output, "w") as f:
        print(script, file=f)


class MindusLexer(Lexer):
    # Set of tokens
    tokens = {
        IF, ELSE, ELIF, WHILE,
        FUNCTION,
        IDENTIFIER, NUMBER_LITERAL, STRING_LITERAL,
        ADD, SUBTRACT, MULTIPLY, DIVIDE, IDIVIDE, MODULUS,
        GREATER_THAN, LESS_THAN, GREATER_THAN_EQ, LESS_THAN_EQ,
        EQUALS, NOT_EQUALS, LOGICAL_AND, LOGICAL_OR,
        BITWISE_AND, BITWISE_XOR, BITWISE_OR,
        ASSIGN, ADD_ASSIGN, SUBTRACT_ASSIGN, MULTIPLY_ASSIGN,
        DIVIDE_ASSIGN, IDIVIDE_ASSIGN, MODULUS_ASSIGN,
        AND_ASSIGN, OR_ASSIGN, XOR_ASSIGN,
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

    ADD             = r'\+=?'
    SUBTRACT        = r'-=?'
    MULTIPLY        = r'\*=?'
    DIVIDE          = r'/=?'
    IDIVIDE         = r'//=?'
    MODULUS         = r'%=?'
    GREATER_THAN_EQ = r'>='
    GREATER_THAN    = r'>'
    LESS_THAN_EQ    = r'<='
    LESS_THAN       = r'<'
    EQUALS          = r'=='
    NOT_EQUALS      = r'!='
    BITWISE_AND     = r'&=?'
    BITWISE_XOR     = r'\^=?'
    BITWISE_OR      = r'\|=?'
    LOGICAL_AND     = r'&&'
    LOGICAL_OR      = r'\|\|'
    ASSIGN          = r'='
    OPEN_PAREN      = r'\('
    CLOSE_PAREN     = r'\)'
    OPEN_BRACE      = r'\{'
    CLOSE_BRACE     = r'\}'
    SEMICOLON       = r';'
    COMMA           = r','

    ADD['+='] = ADD_ASSIGN
    SUBTRACT[ '-='] = SUBTRACT_ASSIGN
    MULTIPLY['*='] = MULTIPLY_ASSIGN
    DIVIDE['/='] = DIVIDE_ASSIGN
    IDIVIDE['//='] = IDIVIDE_ASSIGN
    MODULUS['%='] = MODULUS_ASSIGN
    BITWISE_AND['&='] = AND_ASSIGN
    BITWISE_OR['|='] = OR_ASSIGN
    BITWISE_XOR['^='] = XOR_ASSIGN

    IDENTIFIER['if'] = IF
    IDENTIFIER['else'] = ELSE
    IDENTIFIER['elif'] = ELIF
    IDENTIFIER['while'] = WHILE
    IDENTIFIER['function'] = FUNCTION


class MindusParser(Parser):
    start = 'program'
    tokens = MindusLexer.tokens
    precedence = (
        ('left', ELSE),
        ('right', ASSIGN, ADD_ASSIGN, SUBTRACT_ASSIGN, MULTIPLY_ASSIGN, DIVIDE_ASSIGN,
                  IDIVIDE_ASSIGN, MODULUS_ASSIGN, AND_ASSIGN, OR_ASSIGN, XOR_ASSIGN),
        ('left', LOGICAL_OR),
        ('left', LOGICAL_AND),
        ('left', BITWISE_OR),
        ('left', BITWISE_XOR),
        ('left', BITWISE_AND),
        ('nonassoc', EQUALS, NOT_EQUALS),
        ('nonassoc', GREATER_THAN, LESS_THAN, GREATER_THAN_EQ, LESS_THAN_EQ),
        ('left', ADD, SUBTRACT),
        ('left', MULTIPLY, DIVIDE, IDIVIDE, MODULUS),
    )

    @_('expr MODULUS expr')
    def expr(self, p):
        if isinstance(p[0], NumberNode) and isinstance(p[2], NumberNode):
            return NumberNode(p[0].value % p[2].value)
        return ModulusOperation(p[0], p[2])

    @_('expr IDIVIDE expr')
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

    @_('')
    def parameters(self, p):
        return []

    @_('IDENTIFIER')
    def parameters(self, p):
        return [p[0]]

    @_('parameters COMMA IDENTIFIER')
    def parameters(self, p):
        p[0].append(p[2])
        return p[0]

    @_('IDENTIFIER OPEN_PAREN arguments CLOSE_PAREN')
    def function_call(self, p):
        return FunctionCall(p[0], p[2])

    @_('IDENTIFIER ASSIGN expr')
    def assignment(self, p):
        return Assignment(p[0], p[2])

    @_('IDENTIFIER ADD_ASSIGN expr')
    def assignment(self, p):
        return Assignment(p[0], AddOperation(Variable(p[0]), p[2]))

    @_('IDENTIFIER SUBTRACT_ASSIGN expr')
    def assignment(self, p):
        return Assignment(p[0], SubtractOperation(Variable(p[0]), p[2]))

    @_('IDENTIFIER MULTIPLY_ASSIGN expr')
    def assignment(self, p):
        return Assignment(p[0], MultiplyOperation(Variable(p[0]), p[2]))

    @_('IDENTIFIER DIVIDE_ASSIGN expr')
    def assignment(self, p):
        return Assignment(p[0], DivideOperation(Variable(p[0]), p[2]))

    @_('IDENTIFIER IDIVIDE_ASSIGN expr')
    def assignment(self, p):
        return Assignment(p[0], IntDivideOperation(Variable(p[0]), p[2]))

    @_('IDENTIFIER MODULUS_ASSIGN expr')
    def assignment(self, p):
        return Assignment(p[0], ModulusOperation(Variable(p[0]), p[2]))

    @_('IDENTIFIER AND_ASSIGN expr')
    def assignment(self, p):
        return Assignment(p[0], BitwiseAndOperation(Variable(p[0]), p[2]))

    @_('IDENTIFIER OR_ASSIGN expr')
    def assignment(self, p):
        return Assignment(p[0], BitwiseOrOperation(Variable(p[0]), p[2]))

    @_('IDENTIFIER XOR_ASSIGN expr')
    def assignment(self, p):
        return Assignment(p[0], BitwiseXorOperation(Variable(p[0]), p[2]))

    @_('assignment SEMICOLON', 'function_call SEMICOLON', 'function_def')
    def statement(self, p):
        return p[0]

    @_('statement', 'if_statement', 'while_loop')
    def block(self, p):
        return [p[0]]

    @_('block if_statement', 'block statement', 'block while_loop')
    def block(self, p):
        p[0].append(p[1])
        return p[0]

    @_("FUNCTION IDENTIFIER OPEN_PAREN parameters CLOSE_PAREN OPEN_BRACE block CLOSE_BRACE")
    def function_def(self, p):
        function_name = p[1]
        return DefineFunction(function_name, p.parameters, p.block)

    @_('WHILE OPEN_PAREN expr CLOSE_PAREN OPEN_BRACE block CLOSE_BRACE')
    def while_loop(self, p):
        return Loop(p.expr, p.block)

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

    @_('ELSE if_statement')
    def else_statement(self, p):
        return p.if_statement

    @_('statement', 'if_statement', 'while_loop')
    def program(self, p):
        return Program([p[0]])

    @_('program if_statement', 'program statement', 'program while_loop')
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


class JumpInstruction(Instruction):
    def __init__(self, offset, op='always', left=0, right=0):
        super().__init__()
        self.offset = offset
        self.op = op
        self.left = left
        self.right = right

    def __hash__(self):
        return hash(id(self))

    def emit(self):
        return "jump {} {} {} {}".format(
            self.jump_target.index,
            self.op,
            self.left,
            self.right
        )


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

    def compile(self, state=None):
        if state is None:
            state = {"tokens": ['DISCARD'], "functions": {}, "variables": {}}
        instructions = compile_block(self.units, state)
        instructions.append(ReturnInstruction())
        return instructions


@dataclass
class DefineFunction:
    name: str
    parameters: List[Any]
    block: Any

    def compile(self, state=None):
        state['functions'][self.name] = self
        return []


@dataclass
class FunctionCall:
    name: str
    arguments: List[Any]

    def user_function_compile(self, state):
        function = state['functions'][self.name]
        if len(self.arguments) != len(function.parameters):
            raise ValueError(
                "{} requires {} parameters - received {}".format(
                    self.name,
                    len(function.parameters),
                    len(self.arguments)
                )
            )
        for i in range(len(self.arguments)):
            state['variables'][function.parameters[i]] = self.arguments[i]
        return compile_block(function.block, state)

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
                if arg.name in state['variables']:
                    instructions.append(CallInstruction(
                        'print',
                        '"{}"'.format(state['variables'][arg.name])
                    ))
                else:
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
            if isinstance(arg, Variable):
                arg = state['variables'][arg.name]
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
            if isinstance(arg, Variable):
                arg = state['variables'][arg.name]
            if not isinstance(arg, StringNode):
                raise TypeError("sensor() requires two string literal argumentss")

        token = state['tokens'][-1]
        return [SensorInstruction(
            token,
            self.arguments[0].resolve_string(state),
            "@{}".format(self.arguments[1].resolve_string(state))
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
        elif self.name in state['functions']:
            return self.user_function_compile(state)
        else:
            raise ValueError("Unrecognized function '{}'".format(self.name))


@dataclass
class Assignment:
    identifier: str
    value: Any

    def compile(self, state):
        # If value is a string, store as a string variable
        if isinstance(self.value, StringNode):
            state['variables'][self.identifier] = self.value
            return []
        # If value is a number, skip indirection
        elif isinstance(self.value, NumberNode):
            return [SetInstruction(self.identifier, self.value.value)]
        # If value is a variable reference, skip indirection
        elif isinstance(self.value, Variable):
            return [SetInstruction(self.identifier, self.value.name)]
        # If value is a binary operation where the left side is the same
        # as the target of this assignment, this is an in-place assignment
        elif isinstance(self.value, BinaryOperation) and isinstance(self.value.left, Variable) and self.value.left.name == self.identifier:
            instructions = self.value.compile(state)
            op_instruction = instructions.pop()
            op_instruction.output = self.identifier
            instructions.append(op_instruction)
            return instructions
        # Full indirect assignment
        else:
            with next_token(state) as token:
                instructions = self.value.compile(state)
                instructions.append(SetInstruction(self.identifier, token))
            return instructions


@dataclass
class Loop:
    condition: Any
    block: Any

    def compile(self, state):
        block_instructions = compile_block(self.block, state)

        with next_token(state) as token:
            condition_instructions = self.condition.compile(state)
            # Direct comparison expression
            if isinstance(self.condition, comparisons):
                op_instruction = condition_instructions.pop()
                jump_instruction = JumpInstruction(
                    len(block_instructions) + 2, # Offset
                    inverse_comparisons[op_instruction.op], # Operation
                    op_instruction.left, # Left
                    op_instruction.right, # Right
                )
            # Non-comparison expression, needs to be coerced to bool
            else:
                jump_instruction = JumpInstruction(
                    len(block_instructions) + 2, # Offset
                    "greaterThan", # Operation
                    token, # Left
                    0 # Right
                )


        instructions = []

        # Check to escape the loop
        instructions.extend(condition_instructions)
        instructions.append(jump_instruction)
        # Run the loop body
        instructions.extend(block_instructions)
        # Return to the check
        instructions.append(JumpInstruction(-len(instructions)))

        return instructions


@dataclass
class Conditional:
    condition: Any
    true_block: Any
    false_block: Any

    def compile(self, state):
        true_instructions = compile_block(self.true_block, state)
        if isinstance(self.false_block, Conditional):
            false_instructions = self.false_block.compile(state)
        else:
            false_instructions = compile_block(self.false_block, state)

        with next_token(state) as token:
            condition_instructions = self.condition.compile(state)

            # Direct comparison expression
            if isinstance(self.condition, comparisons):
                op_instruction = condition_instructions.pop()
                jump_instruction = JumpInstruction(
                    len(false_instructions) + 2, # Offset
                    inverse_comparisons[op_instruction.op], # Operation
                    op_instruction.left, # Left
                    op_instruction.right, # Right
                )
            # Non-comparison expression, needs to be coerced to bool
            else:
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
        instructions.append(JumpInstruction(len(true_instructions) + 1))
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

    def resolve_string(self, state):
        return self.value


@dataclass
class Variable:
    name: str

    def compile(self, state):
        token = state['tokens'][-1]
        return [SetInstruction(token, self.name)]

    def resolve_string(self, state):
        return state['variables'][self.name].value


@dataclass
class BinaryOperation:
    left: Any
    right: Any

    def compile(self, state):
        token = state['tokens'][-1]
        instructions = []

        with next_token(state) as left_token:
            # If left is a number, set it as the left operand
            if isinstance(self.left, NumberNode):
                left_token = self.left.value
            # If left is a variable, set it as the left operand
            elif isinstance(self.left, Variable):
                left_token = self.left.name
            # If left is a subexpression, use a full sub-compile
            else:
                instructions.extend(self.left.compile(state))

            with next_token(state) as right_token:
                # If right is a number, set it as the right operand
                if isinstance(self.right, NumberNode):
                    right_token = self.right.value
                # If right is a variable, set it as the right operand
                elif isinstance(self.right, Variable):
                    right_token = self.right.name
                # If right is a subexpression, use a full sub-compile
                else:
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


comparisons = (
    EqualCompare, NotEqualCompare,
    LessThanCompare, LessThanEqCompare,
    GreaterThanCompare, GreaterThanEqCompare,
)


inverse_comparisons = {
    'equal': 'notEqual',
    'notEqual': 'equal',
    'lessThan': 'greaterThanEq',
    'greaterThanEq': 'lessThan',
    'greaterThan': 'lessThanEq',
    'lessThanEq': 'greaterThan'
}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('source', nargs='+', type=Path)
    parser.add_argument('-o', '--output', default=None)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    if args.output is None:
        args.output = args.source[0].with_suffix('.mlog')

    main(args.source, args.output, args.verbose)
