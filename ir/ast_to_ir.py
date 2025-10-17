# ir/ast_to_ir.py
from typing import Dict, Any
from frontend.ast import Node, AssignNode, BinOpNode, VarNode, NumNode, UnaryOpNode, CallNode, IndexNode
from ir.ir import (
    IRModule, IRFunction, IRBuilder, IRValue, IRType, IROpCode, IRBlock,
    print_ir_module
)

class ASTToIRVisitor:
    """Visitor to convert AST to IR."""
    
    def __init__(self, module: IRModule):
        self.module = module
        self.builder = IRBuilder(module)
        self.current_function: IRFunction = None
        
    def visit_assign(self, node: AssignNode) -> IRValue:
        """Convert assignment node to IR."""
        # Generate IR for the expression
        expr_result = node.expr.accept(self)
        
        # Get or create the variable
        var = self.current_function.get_variable(node.var_name)
        
        # Store the result
        self.builder.emit_store(var, expr_result)
        
        return var
    
    def visit_binop(self, node: BinOpNode) -> IRValue:
        """Convert binary operation to IR."""
        left = node.left.accept(self)
        right = node.right.accept(self)
        
        # Map AST operations to IR opcodes
        op_map = {
            '+': IROpCode.ADD,
            '-': IROpCode.SUB,
            '*': IROpCode.MUL,
            '/': IROpCode.DIV,
            '@': IROpCode.MUL  # Matrix multiplication - treated as MUL for now
        }
        
        if node.op not in op_map:
            raise ValueError(f"Unsupported binary operation: {node.op}")
        
        return self.builder.emit_binary_op(op_map[node.op], left, right)
    
    def visit_var(self, node: VarNode) -> IRValue:
        """Convert variable reference to IR."""
        var = self.current_function.get_variable(node.name)
        return self.builder.emit_load(var)
    
    def visit_num(self, node: NumNode) -> IRValue:
        """Convert number literal to IR."""
        return self.current_function.create_constant(node.value)
    
    def visit_unary(self, node: UnaryOpNode) -> IRValue:
        """Convert unary operation to IR."""
        operand = node.operand.accept(self)
        
        if node.op == '-':
            return self.builder.emit_unary_op(IROpCode.NEG, operand)
        elif node.op == '+':
            return operand  # Unary plus is identity
        else:
            raise ValueError(f"Unsupported unary operation: {node.op}")
    
    def visit_call(self, node: CallNode) -> IRValue:
        """Convert function call to IR."""
        # Handle mathematical functions
        math_functions = {
            'sin': IROpCode.SIN,
            'cos': IROpCode.COS,
            'tan': IROpCode.TAN,
            'sqrt': IROpCode.SQRT,
            'exp': IROpCode.EXP,
            'log': IROpCode.LOG
        }
        
        if node.func_name in math_functions:
            if len(node.args) != 1:
                raise ValueError(f"Function {node.func_name} expects 1 argument, got {len(node.args)}")
            
            arg = node.args[0].accept(self)
            return self.builder.emit_unary_op(math_functions[node.func_name], arg)
        
        elif node.func_name == 'pow':
            if len(node.args) != 2:
                raise ValueError(f"Function pow expects 2 arguments, got {len(node.args)}")
            
            base = node.args[0].accept(self)
            exponent = node.args[1].accept(self)
            return self.builder.emit_binary_op(IROpCode.POW, base, exponent)
        
        else:
            raise ValueError(f"Unsupported function: {node.func_name}")
    
    def visit_index(self, node: IndexNode) -> IRValue:
        """Convert array indexing to IR."""
        # For now, treat as a simple load operation
        # In a more complete implementation, this would handle array bounds checking
        array = node.array.accept(self)
        index = node.index.accept(self)
        
        result = self.current_function.create_temp()
        self.builder.emit_instruction(IROpCode.LOAD, result, [array, index])
        return result

def convert_ast_to_ir(ast_node: Node, module_name: str = "main") -> IRModule:
    """Convert an AST to IR representation."""
    module = IRModule(module_name)
    visitor = ASTToIRVisitor(module)
    
    # Create a main function
    func = visitor.builder.build_function("main")
    visitor.current_function = func
    
    # Convert the AST
    result = ast_node.accept(visitor)
    
    # Add a return statement if needed
    if not func.blocks[-1].instructions or func.blocks[-1].instructions[-1].opcode != IROpCode.STORE:
        visitor.builder.emit_instruction(IROpCode.ASSIGN, None, [result])
    
    return module

def optimize_ir_basic(module: IRModule) -> IRModule:
    """Basic IR optimizations - constant folding and dead code elimination."""
    
    for func in module.functions.values():
        for block in func.blocks:
            new_instructions = []
            
            for instr in block.instructions:
                # Constant folding for binary operations
                if (instr.opcode in [IROpCode.ADD, IROpCode.SUB, IROpCode.MUL, IROpCode.DIV] and
                    len(instr.operands) == 2 and
                    all(op.is_constant for op in instr.operands)):
                    
                    left_val = instr.operands[0].value
                    right_val = instr.operands[1].value
                    
                    if instr.opcode == IROpCode.ADD:
                        result_val = left_val + right_val
                    elif instr.opcode == IROpCode.SUB:
                        result_val = left_val - right_val
                    elif instr.opcode == IROpCode.MUL:
                        result_val = left_val * right_val
                    elif instr.opcode == IROpCode.DIV:
                        if right_val == 0:
                            new_instructions.append(instr)  # Keep division by zero for error handling
                            continue
                        result_val = left_val / right_val
                    
                    # Replace with constant
                    if instr.result:
                        instr.result.is_constant = True
                        instr.result.value = result_val
                        # Remove the instruction since the result is now a constant
                        continue
                
                # Algebraic simplifications
                elif (instr.opcode == IROpCode.ADD and len(instr.operands) == 2):
                    left, right = instr.operands
                    
                    # x + 0 = x
                    if right.is_constant and right.value == 0:
                        instr.opcode = IROpCode.ASSIGN
                        instr.operands = [left]
                    # 0 + x = x
                    elif left.is_constant and left.value == 0:
                        instr.opcode = IROpCode.ASSIGN
                        instr.operands = [right]
                
                elif (instr.opcode == IROpCode.MUL and len(instr.operands) == 2):
                    left, right = instr.operands
                    
                    # x * 0 = 0
                    if (right.is_constant and right.value == 0) or (left.is_constant and left.value == 0):
                        if instr.result:
                            instr.result.is_constant = True
                            instr.result.value = 0
                            continue
                    # x * 1 = x
                    elif right.is_constant and right.value == 1:
                        instr.opcode = IROpCode.ASSIGN
                        instr.operands = [left]
                    # 1 * x = x
                    elif left.is_constant and left.value == 1:
                        instr.opcode = IROpCode.ASSIGN
                        instr.operands = [right]
                
                new_instructions.append(instr)
            
            block.instructions = new_instructions
    
    return module