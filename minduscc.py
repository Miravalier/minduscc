#!/usr/bin/env python3
from dataclasses import dataclass
from sly import Lexer, Parser
from typing import List, Any


class MindusLexer(Lexer):
    # Set of tokens
    tokens = {
        IF,
        THEN,
        ELSE,
        ENDIF,
        IDENTIFIER,
        NUMBER_LITERAL, STRING_LITERAL,
        ADD, SUBTRACT, MULTIPLY, DIVIDE, INT_DIVIDE, MODULUS,
        GREATER_THAN, LESS_THAN, GREATER_THAN_EQ, LESS_THAN_EQ,
        EQUALS, NOT_EQUALS, LOGICAL_AND, LOGICAL_OR,
        BITWISE_AND, BITWISE_XOR, BITWISE_OR,
        ASSIGN,
        OPEN_PAREN, CLOSE_PAREN,
        MEMBER,
        SEMICOLON,
    }

    # Characters ignored between tokens
    ignore = ' \t'

    @_(r'\n+')
    def ignore_newline(self, t):
        self.lineno += len(t.value)

    # Regex Rules for tokens
    IDENTIFIER      = r'[a-zA-Z_][a-zA-Z0-9_]*'
    NUMBER_LITERAL  = r'\d+|0x\d+'
    STRING_LITERAL  = r'"[^"]*"' + '|' + r"'[^']*'"
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
    MEMBER          = r'\.'
    SEMICOLON       = r';'

    IDENTIFIER['if'] = IF
    IDENTIFIER['then'] = THEN
    IDENTIFIER['else'] = ELSE
    IDENTIFIER['endif'] = ENDIF


@dataclass
class Program:
    statements: List[Any]


@dataclass
class FunctionCall:
    name: str
    arguments: List[Any]


@dataclass
class GetMember:
    name: str
    member: str


@dataclass
class Assignment:
    identifier: str
    value: Any


@dataclass
class Conditional:
    condition: Any
    true_statement: Any
    false_statement: Any


@dataclass
class NumberNode:
    value: int


@dataclass
class StringNode:
    value: str


@dataclass
class Variable:
    name: str


@dataclass
class AddOperation:
    left: Any
    right: Any


@dataclass
class SubtractOperation:
    left: Any
    right: Any


@dataclass
class MultiplyOperation:
    left: Any
    right: Any


@dataclass
class DivideOperation:
    left: Any
    right: Any


@dataclass
class GreaterThanCompare:
    left: Any
    right: Any


@dataclass
class GreaterThanEqCompare:
    left: Any
    right: Any


@dataclass
class LessThanCompare:
    left: Any
    right: Any


@dataclass
class LessThanEqCompare:
    left: Any
    right: Any


@dataclass
class EqualCompare:
    left: Any
    right: Any


@dataclass
class NotEqualCompare:
    left: Any
    right: Any


class MindusParser(Parser):
    start = 'program'
    tokens = MindusLexer.tokens
    precedence = (
        ('nonassoc', ASSIGN),
        ('nonassoc', EQUALS, NOT_EQUALS),
        ('nonassoc', GREATER_THAN, LESS_THAN, GREATER_THAN_EQ, LESS_THAN_EQ),
        ('left', ADD, SUBTRACT),
        ('left', MULTIPLY, DIVIDE),
        ('right', MEMBER),
    )

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
        return NumberNode(int(p[0]))

    @_('STRING_LITERAL')
    def expr(self, p):
        return StringNode(p[0])

    @_('IDENTIFIER')
    def expr(self, p):
        return Variable(p[0])

    @_('function_call')
    def expr(self, p):
        return p[0]

    @_('IDENTIFIER MEMBER IDENTIFIER')
    def expr(self, p):
        return GetMember(p[0], p[2])

    @_('expr')
    def arguments(self, p):
        return [p[0]]

    @_('arguments expr')
    def arguments(self, p):
        p[0].append(p[1])
        return p[0]

    @_('IDENTIFIER OPEN_PAREN arguments CLOSE_PAREN')
    def function_call(self, p):
        return FunctionCall(p[0], p[2])

    @_('IDENTIFIER MEMBER function_call')
    def function_call(self, p):
        p[2].name = GetMember(p[0], p[2].name)
        return p[2]

    @_('IDENTIFIER ASSIGN expr')
    def assignment(self, p):
        return Assignment(p[0], p[2])

    @_('assignment SEMICOLON', 'function_call SEMICOLON')
    def statement(self, p):
        return p[0]

    @_('statement')
    def block(self, p):
        return [p[0]]

    @_('block statement')
    def block(self, p):
        p[0].append(p[1])
        return p[0]

    @_('IF expr THEN block ENDIF')
    def if_statement(self, p):
        return Conditional(p.expr, p.block, None)

    @_('IF expr THEN block ELSE block ENDIF')
    def if_statement(self, p):
        return Conditional(p.expr, p.block0, p.block1)

    @_('statement', 'if_statement')
    def program(self, p):
        return Program([p[0]])

    @_('program if_statement', 'program statement')
    def program(self, p):
        p.program.statements.append(p[1])
        return p.program


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('source', nargs='+')
    parser.add_argument('-o', '--output', required=True)
    args = parser.parse_args()

    lexer = MindusLexer()
    parser = MindusParser()

    for src in args.source:
        with open(src, "r") as f:
            raw = f.read()
        tokens = lexer.tokenize(raw)
        ast = parser.parse(tokens)
        print(ast)

