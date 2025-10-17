# optimizer/optimize.py
from typing import List, Dict, Set, Optional
from ir.ir import (
    IRModule, IRFunction, IRBlock, IRInstruction, IRValue, IROpCode,
    print_ir_module
)
from ir.ast_to_ir import optimize_ir_basic

class IROptimizer:
    """Advanced IR optimization passes."""
    
    def __init__(self):
        self.use_counts: Dict[str, int] = {}
        self.def_counts: Dict[str, int] = {}
    
    def optimize_module(self, module: IRModule) -> IRModule:
        """Apply all optimization passes to a module."""
        # Basic optimizations first
        module = optimize_ir_basic(module)
        
        # Advanced optimizations
        for func in module.functions.values():
            self.optimize_function(func)
        
        return module
    
    def optimize_function(self, func: IRFunction):
        """Optimize a single function."""
        # Build use-def chains
        self._build_use_def_chains(func)
        
        # Apply optimizations
        self._eliminate_dead_code(func)
        self._common_subexpression_elimination(func)
        self._strength_reduction(func)
        self._loop_invariant_code_motion(func)
    
    def _build_use_def_chains(self, func: IRFunction):
        """Build use-definition chains for analysis."""
        self.use_counts.clear()
        self.def_counts.clear()
        
        for block in func.blocks:
            for instr in block.instructions:
                # Count definitions
                if instr.result and not instr.result.is_constant:
                    self.def_counts[instr.result.name] = self.def_counts.get(instr.result.name, 0) + 1
                
                # Count uses
                for operand in instr.operands:
                    if not operand.is_constant:
                        self.use_counts[operand.name] = self.use_counts.get(operand.name, 0) + 1
    
    def _eliminate_dead_code(self, func: IRFunction):
        """Remove instructions that compute unused values."""
        for block in func.blocks:
            new_instructions = []
            
            for instr in block.instructions:
                # Keep instruction if:
                # 1. It has side effects (STORE, CALL)
                # 2. Its result is used
                # 3. It's a control flow instruction
                keep = True
                
                if instr.opcode in [IROpCode.STORE, IROpCode.CALL]:
                    keep = True  # Has side effects
                elif instr.result and not instr.result.is_temp:
                    keep = True  # User variables should be kept
                elif instr.result and instr.result.is_temp:
                    # Temporary - only keep if used
                    keep = self.use_counts.get(instr.result.name, 0) > 0
                
                if keep:
                    new_instructions.append(instr)
            
            block.instructions = new_instructions
    
    def _common_subexpression_elimination(self, func: IRFunction):
        """Eliminate common subexpressions."""
        # Track expressions by their "signature"
        expr_map: Dict[str, IRValue] = {}
        
        for block in func.blocks:
            block_expr_map = expr_map.copy()  # Local copy for this block
            new_instructions = []
            
            for instr in block.instructions:
                # Create signature for arithmetic operations
                if instr.opcode in [IROpCode.ADD, IROpCode.SUB, IROpCode.MUL, IROpCode.DIV]:
                    if len(instr.operands) == 2:
                        left, right = instr.operands
                        
                        # Commutative operations: normalize order
                        if instr.opcode in [IROpCode.ADD, IROpCode.MUL]:
                            if left.name > right.name:  # Lexicographic ordering
                                left, right = right, left
                        
                        signature = f"{instr.opcode.value}_{left.name}_{right.name}"
                        
                        if signature in block_expr_map:
                            # Replace with existing result
                            existing_result = block_expr_map[signature]
                            if instr.result:
                                # Create assignment instead
                                assign_instr = IRInstruction(
                                    IROpCode.ASSIGN,
                                    instr.result,
                                    [existing_result]
                                )
                                new_instructions.append(assign_instr)
                            continue
                        else:
                            # Record this expression
                            if instr.result:
                                block_expr_map[signature] = instr.result
                
                new_instructions.append(instr)
            
            block.instructions = new_instructions
    
    def _strength_reduction(self, func: IRFunction):
        """Replace expensive operations with cheaper equivalents."""
        for block in func.blocks:
            new_instructions = []
            
            for instr in block.instructions:
                # x * 2 -> x + x
                if (instr.opcode == IROpCode.MUL and len(instr.operands) == 2):
                    left, right = instr.operands
                    
                    # Check for multiplication by power of 2
                    if right.is_constant and right.value == 2:
                        # Replace with addition
                        new_instr = IRInstruction(
                            IROpCode.ADD,
                            instr.result,
                            [left, left],
                            instr.metadata
                        )
                        new_instructions.append(new_instr)
                        continue
                    elif left.is_constant and left.value == 2:
                        # Replace with addition
                        new_instr = IRInstruction(
                            IROpCode.ADD,
                            instr.result,
                            [right, right],
                            instr.metadata
                        )
                        new_instructions.append(new_instr)
                        continue
                
                # x / 2 -> x * 0.5 (if division by constant)
                elif (instr.opcode == IROpCode.DIV and len(instr.operands) == 2):
                    left, right = instr.operands
                    
                    if right.is_constant and right.value != 0:
                        # Replace with multiplication by reciprocal
                        reciprocal = func.create_constant(1.0 / right.value)
                        new_instr = IRInstruction(
                            IROpCode.MUL,
                            instr.result,
                            [left, reciprocal],
                            instr.metadata
                        )
                        new_instructions.append(new_instr)
                        continue
                
                new_instructions.append(instr)
            
            block.instructions = new_instructions
    
    def _loop_invariant_code_motion(self, func: IRFunction):
        """Move loop-invariant code out of loops (simplified version)."""
        # For now, just identify potentially invariant operations
        # In a full implementation, this would require loop analysis
        
        invariant_candidates = set()
        
        for block in func.blocks:
            for instr in block.instructions:
                # Operations with only constant operands are loop invariant
                if (instr.opcode in [IROpCode.ADD, IROpCode.SUB, IROpCode.MUL, IROpCode.DIV] and
                    all(op.is_constant for op in instr.operands)):
                    invariant_candidates.add(instr)
        
        # For now, just mark them in metadata
        for instr in invariant_candidates:
            instr.metadata['loop_invariant'] = True

def optimize_cuda_specific(module: IRModule) -> IRModule:
    """CUDA-specific optimizations."""
    
    for func in module.functions.values():
        for block in func.blocks:
            new_instructions = []
            
            for instr in block.instructions:
                # Vectorization opportunities
                if instr.opcode in [IROpCode.ADD, IROpCode.MUL]:
                    instr.metadata['vectorizable'] = True
                
                # Memory coalescing hints
                if instr.opcode == IROpCode.LOAD:
                    instr.metadata['memory_pattern'] = 'sequential'  # Assume sequential for now
                
                # Register pressure hints
                if instr.result and instr.result.is_temp:
                    instr.metadata['register_pressure'] = 'low'
                
                new_instructions.append(instr)
            
            block.instructions = new_instructions
    
    return module

def optimize(ast_or_ir, target='cuda'):
    """Main optimization entry point - supports both AST and IR."""
    
    # If it's AST, convert to IR first
    if hasattr(ast_or_ir, 'var_name'):  # Duck typing for AST
        from ir.ast_to_ir import convert_ast_to_ir
        module = convert_ast_to_ir(ast_or_ir)
    else:
        module = ast_or_ir
    
    # Apply general optimizations
    optimizer = IROptimizer()
    optimized_module = optimizer.optimize_module(module)
    
    # Apply target-specific optimizations
    if target == 'cuda':
        optimized_module = optimize_cuda_specific(optimized_module)
    
    return optimized_module
