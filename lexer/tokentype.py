from enum import Enum, auto


class TokenType(Enum):
    LEFT_PAREN = auto()
    RIGHT_PAREN = auto()
    LEFT_BRACE = auto()
    RIGHT_BRACE = auto()
    LEFT_BRACKET = auto()
    RIGHT_BRACKET = auto()
    SEMICOLON = auto()
    COLON = auto()

    COMMA = auto()
    DOT = auto()
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    MODULAR = auto()

    EQUAL = auto()
    EQUAL_EQUAL = auto()
    EXCLAM = auto()
    EXCLAM_EQUAL = auto()
    GREATER = auto()
    GREATER_EQUAL = auto()
    LESS = auto()
    LESS_EQUAL = auto()

    IDENTIFIER = auto()

    BOOLEAN = auto()
    NUMBER = auto()
    SINGLE_STRING = auto()
    DOUBLE_STRING = auto()

    EOF = auto()
    UNKNOWN = auto()
    TAB = '\t'

    IF = 'if'
    ELSE = 'else'
    AND = 'and'
    OR = 'or'
    FUN = 'def'
    VAR = 'var'
    TRUE = 'true'
    FALSE = 'false'
    NULL = 'null'
    FOR = 'for'
    WHILE = 'while'
    IN = 'in'
    RETURN = 'return'
    SWITCH = 'switch'
    CASE = 'case'
    BREAK = 'break'
    CONTINUE = 'continue'
    NEW = 'new'
    CLASS = 'class'
    INTERFACE = 'interface'
    EXTENDS = 'extends'
    IMPLEMENT = 'implement'
    PACKAGE = 'package'
    IMPORT = 'import'
    NOT = 'not'

    STATIC = 'static'

    PUBLIC = 'public'
    PROTECTED = 'protected'
    DEFAULT = 'default'
    PRIVATE = 'private'
    VOID = 'void'
    THROWS = 'throws'
    THROW = 'throw'
    INT = 'int'
    THIS = 'this'
    BYTE = 'byte'
    CHAR = 'char'
    DO = 'do'
    DOUBLE = 'double'
    FINALLY = 'finally'
    FLOAT = 'float'
    INSTANCEOF = 'instanceof'
    LONG = 'long'
    NATIVE = 'native'
    SHORT = 'short'
    SUPER = 'super'
    SYNCHRONIZED = 'synchronized'
    ENUM = 'enum'
    TRANSIENT = 'transient'
    VOLATILE = 'volatile'
    STRICTFP = 'strictfp'
    ASSERT = 'assert'

    ### C ###
    AUTO = 'auto'
    STRUCT = 'struct'
    ASM = 'asm'
    CONST = 'const'
    EXTERN = 'extern'
    GOTO = 'goto'
    REGISTER = 'register'
    SIGNED = 'signed'
    SIZEOF = 'sizeof'
    TYPEDEF = 'typedef'
    UNION = 'union'
    UNSIGNED = 'unsigned'
    DEFINE = '#define'
    INCLUDE = '#include'
    ELIF = 'elif'
    ENDIF = 'endif'
    ERROR = 'error'
    IFDEF = '#ifdef'
    INCLUDE_NEXT = 'include_next'
    LINE = 'line'
    PRAGMA = 'pragma'
    UNDEF = 'undef'
    INT32 = 'int32'
    INT64 = 'int64'
    INT8 = 'int8'
    INTERRUPT = 'interrupt'
    LEAVE = 'leave'
    NEAR = 'near'

    # PRINT = 'print'

    @classmethod
    def has_value(cls, value):
        return any(value == item.value for item in cls)
