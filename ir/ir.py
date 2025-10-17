# ir/ir.py
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod

class IRType(Enum):
    FLOAT = "float"
    INT = "int"
    VECTOR = "vector"
    MATRIX = "matrix"

class IROpCode(Enum):
    # Arithmetic
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    NEG = "neg"
    
    # Memory
    LOAD = "load"
    STORE = "store"
    ALLOC = "alloc"
    
    # Math functions
    SIN = "sin"
    COS = "cos"
    TAN = "tan"
    SQRT = "sqrt"
    EXP = "exp"
    LOG = "log"
    POW = "pow"
    
    # Control
    ASSIGN = "assign"
    CALL = "call"
    
    # Constants
    CONST = "const"

@dataclass
class IRValue:
    """Represents a value in the IR - can be temporary, variable, or constant."""
    name: str
    type: IRType
    is_temp: bool = False
    is_constant: bool = False
    value: Optional[Any] = None
    
    def __post_init__(self):
        if self.is_constant and self.value is None:
            raise ValueError("Constant values must have a value")

@dataclass
class IRInstruction:
    """Single instruction in the IR."""
    opcode: IROpCode
    result: Optional[IRValue]
    operands: List[IRValue]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class IRBlock:
    """Basic block in the IR."""
    def __init__(self, name: str):
        self.name = name
        self.instructions: List[IRInstruction] = []
        self.predecessors: List['IRBlock'] = []
        self.successors: List['IRBlock'] = []
    
    def add_instruction(self, instruction: IRInstruction):
        self.instructions.append(instruction)
    
    def __repr__(self):
        return f"IRBlock({self.name}, {len(self.instructions)} instructions)"

class IRFunction:
    """Represents a function in the IR."""
    def __init__(self, name: str, return_type: IRType = IRType.FLOAT):
        self.name = name
        self.return_type = return_type
        self.parameters: List[IRValue] = []
        self.blocks: List[IRBlock] = []
        self.entry_block: Optional[IRBlock] = None
        self.local_variables: Dict[str, IRValue] = {}
        self.temp_counter = 0
    
    def create_temp(self, type: IRType = IRType.FLOAT) -> IRValue:
        """Create a new temporary variable."""
        temp_name = f"t{self.temp_counter}"
        self.temp_counter += 1
        return IRValue(temp_name, type, is_temp=True)
    
    def create_constant(self, value: Any, type: IRType = IRType.FLOAT) -> IRValue:
        """Create a constant value."""
        const_name = f"c_{hash(value) & 0xFFFF:04x}"
        return IRValue(const_name, type, is_constant=True, value=value)
    
    def add_block(self, block: IRBlock):
        self.blocks.append(block)
        if self.entry_block is None:
            self.entry_block = block
    
    def get_variable(self, name: str, type: IRType = IRType.FLOAT) -> IRValue:
        """Get or create a variable."""
        if name not in self.local_variables:
            self.local_variables[name] = IRValue(name, type)
        return self.local_variables[name]

class IRModule:
    """Top-level IR module containing functions and global data."""
    def __init__(self, name: str = "module"):
        self.name = name
        self.functions: Dict[str, IRFunction] = {}
        self.global_variables: Dict[str, IRValue] = {}
    
    def add_function(self, func: IRFunction):
        self.functions[func.name] = func
    
    def get_function(self, name: str) -> Optional[IRFunction]:
        return self.functions.get(name)

# AST to IR conversion
class IRBuilder:
    """Builds IR from AST nodes."""
    
    def __init__(self, module: IRModule):
        self.module = module
        self.current_function: Optional[IRFunction] = None
        self.current_block: Optional[IRBlock] = None
    
    def build_function(self, name: str, return_type: IRType = IRType.FLOAT) -> IRFunction:
        """Create and set a new function as current."""
        func = IRFunction(name, return_type)
        self.module.add_function(func)
        self.current_function = func
        
        # Create entry block
        entry_block = IRBlock("entry")
        func.add_block(entry_block)
        self.current_block = entry_block
        
        return func
    
    def emit_instruction(self, opcode: IROpCode, result: Optional[IRValue] = None, 
                        operands: List[IRValue] = None, **metadata) -> IRInstruction:
        """Emit an instruction to the current block."""
        if operands is None:
            operands = []
        instruction = IRInstruction(opcode, result, operands, metadata)
        if self.current_block:
            self.current_block.add_instruction(instruction)
        return instruction
    
    def emit_binary_op(self, opcode: IROpCode, left: IRValue, right: IRValue) -> IRValue:
        """Emit a binary operation and return the result."""
        result = self.current_function.create_temp()
        self.emit_instruction(opcode, result, [left, right])
        return result
    
    def emit_unary_op(self, opcode: IROpCode, operand: IRValue) -> IRValue:
        """Emit a unary operation and return the result."""
        result = self.current_function.create_temp()
        self.emit_instruction(opcode, result, [operand])
        return result
    
    def emit_load(self, var: IRValue) -> IRValue:
        """Emit a load instruction."""
        if var.is_constant:
            return var  # Constants don't need loading
        result = self.current_function.create_temp()
        self.emit_instruction(IROpCode.LOAD, result, [var])
        return result
    
    def emit_store(self, var: IRValue, value: IRValue):
        """Emit a store instruction."""
        self.emit_instruction(IROpCode.STORE, None, [var, value])

def print_ir_function(func: IRFunction, indent: str = ""):
    """Pretty print an IR function."""
    print(f"{indent}function {func.name}() -> {func.return_type.value}:")
    
    for param in func.parameters:
        print(f"{indent}  param {param.name}: {param.type.value}")
    
    for var_name, var in func.local_variables.items():
        if not var.is_temp:
            print(f"{indent}  var {var_name}: {var.type.value}")
    
    for block in func.blocks:
        print(f"{indent}  {block.name}:")
        for instr in block.instructions:
            operands_str = ", ".join(op.name for op in instr.operands)
            if instr.result:
                print(f"{indent}    {instr.result.name} = {instr.opcode.value} {operands_str}")
            else:
                print(f"{indent}    {instr.opcode.value} {operands_str}")

def print_ir_module(module: IRModule):
    """Pretty print an IR module."""
    print(f"module {module.name}:")
    
    for global_name, global_var in module.global_variables.items():
        print(f"  global {global_name}: {global_var.type.value}")
    
    for func_name, func in module.functions.items():
        print_ir_function(func, "  ")