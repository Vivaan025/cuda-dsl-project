# codegen/ir_to_cuda.py
from typing import Dict, List, Set
from ir.ir import IRModule, IRFunction, IRBlock, IRInstruction, IRValue, IROpCode, IRType
from jinja2 import Environment, FileSystemLoader
import os

class IRToCUDAGenerator:
    """Generate CUDA code from optimized IR."""
    
    def __init__(self, template_dir: str = None):
        if template_dir is None:
            template_dir = os.path.join(os.path.dirname(__file__), '../templates')
        self.env = Environment(loader=FileSystemLoader(template_dir))
        self.temp_counter = 0
    
    def generate_cuda_kernel(self, module: IRModule, kernel_name: str = "kernel") -> str:
        """Generate a complete CUDA kernel from IR module."""
        
        if not module.functions:
            raise ValueError("Module has no functions")
        
        main_func = list(module.functions.values())[0]  # Get first function
        
        # Analyze the function to determine kernel parameters
        input_vars, output_vars, temp_vars = self._analyze_function(main_func)
        
        # Generate CUDA code for each basic block
        cuda_body = self._generate_function_body(main_func)
        
        # Generate complete kernel
        template = self.env.get_template('optimized_kernel.cu.j2')
        return template.render(
            kernel_name=kernel_name,
            input_vars=list(input_vars),
            output_vars=list(output_vars),
            temp_vars=list(temp_vars),
            body=cuda_body
        )
    
    def _analyze_function(self, func: IRFunction):
        """Analyze function to identify input/output/temporary variables."""
        input_vars = set()
        output_vars = set()
        temp_vars = set()
        
        for block in func.blocks:
            for instr in block.instructions:
                # Analyze result
                if instr.result:
                    if instr.result.is_temp:
                        temp_vars.add(instr.result.name)
                    else:
                        output_vars.add(instr.result.name)
                
                # Analyze operands
                for operand in instr.operands:
                    if not operand.is_constant and not operand.is_temp:
                        # Check if it's being stored to (output) or loaded from (input)
                        if instr.opcode == IROpCode.STORE and operand == instr.operands[0]:
                            output_vars.add(operand.name)
                        else:
                            input_vars.add(operand.name)
        
        # Remove outputs from inputs
        input_vars -= output_vars
        
        # Sort temp vars to ensure consistent ordering
        temp_vars = sorted(temp_vars)
        
        return input_vars, output_vars, temp_vars
    
    def _generate_function_body(self, func: IRFunction) -> str:
        """Generate CUDA code for function body."""
        lines = []
        
        for block in func.blocks:
            if len(func.blocks) > 1:  # Only add labels if multiple blocks
                lines.append(f"{block.name}:")
            
            for instr in block.instructions:
                cuda_line = self._generate_instruction(instr)
                if cuda_line:
                    lines.append(f"    {cuda_line}")
        
        return '\n'.join(lines)
    
    def _generate_instruction(self, instr: IRInstruction) -> str:
        """Generate CUDA code for a single instruction."""
        
        if instr.opcode == IROpCode.ADD:
            left, right = instr.operands
            left_code = self._generate_value_access(left)
            right_code = self._generate_value_access(right)
            result_code = self._generate_value_access(instr.result)
            return f"{result_code} = {left_code} + {right_code};"
        
        elif instr.opcode == IROpCode.SUB:
            left, right = instr.operands
            left_code = self._generate_value_access(left)
            right_code = self._generate_value_access(right)
            result_code = self._generate_value_access(instr.result)
            return f"{result_code} = {left_code} - {right_code};"
        
        elif instr.opcode == IROpCode.MUL:
            left, right = instr.operands
            left_code = self._generate_value_access(left)
            right_code = self._generate_value_access(right)
            result_code = self._generate_value_access(instr.result)
            
            # Check for vectorization hint
            if instr.metadata.get('vectorizable'):
                return f"{result_code} = {left_code} * {right_code}; // vectorizable"
            else:
                return f"{result_code} = {left_code} * {right_code};"
        
        elif instr.opcode == IROpCode.DIV:
            left, right = instr.operands
            left_code = self._generate_value_access(left)
            right_code = self._generate_value_access(right)
            result_code = self._generate_value_access(instr.result)
            return f"{result_code} = {left_code} / {right_code};"
        
        elif instr.opcode == IROpCode.NEG:
            operand = instr.operands[0]
            operand_code = self._generate_value_access(operand)
            result_code = self._generate_value_access(instr.result)
            return f"{result_code} = -{operand_code};"
        
        elif instr.opcode == IROpCode.LOAD:
            if len(instr.operands) == 1:  # Simple load
                operand = instr.operands[0]
                operand_code = self._generate_value_access(operand, is_load=True)
                result_code = self._generate_value_access(instr.result)
                
                # Add memory coalescing hints
                if instr.metadata.get('memory_pattern') == 'sequential':
                    return f"{result_code} = {operand_code}; // coalesced access"
                else:
                    return f"{result_code} = {operand_code};"
            else:  # Array access
                array, index = instr.operands
                array_code = self._generate_value_access(array)
                index_code = self._generate_value_access(index)
                result_code = self._generate_value_access(instr.result)
                return f"{result_code} = {array_code}[{index_code}];"
        
        elif instr.opcode == IROpCode.STORE:
            var, value = instr.operands
            var_code = self._generate_value_access(var, is_store=True)
            value_code = self._generate_value_access(value)
            return f"{var_code} = {value_code};"
        
        elif instr.opcode == IROpCode.ASSIGN:
            value = instr.operands[0]
            value_code = self._generate_value_access(value)
            result_code = self._generate_value_access(instr.result)
            return f"{result_code} = {value_code};"
        
        elif instr.opcode in [IROpCode.SIN, IROpCode.COS, IROpCode.TAN, 
                              IROpCode.SQRT, IROpCode.EXP, IROpCode.LOG]:
            operand = instr.operands[0]
            operand_code = self._generate_value_access(operand)
            result_code = self._generate_value_access(instr.result)
            func_name = instr.opcode.value
            return f"{result_code} = {func_name}f({operand_code});"
        
        elif instr.opcode == IROpCode.POW:
            base, exponent = instr.operands
            base_code = self._generate_value_access(base)
            exp_code = self._generate_value_access(exponent)
            result_code = self._generate_value_access(instr.result)
            return f"{result_code} = powf({base_code}, {exp_code});"
        
        else:
            return f"// Unsupported instruction: {instr.opcode.value}"
    
    def _generate_value_access(self, value: IRValue, is_load: bool = False, is_store: bool = False) -> str:
        """Generate CUDA code to access a value."""
        
        if value.is_constant:
            if isinstance(value.value, float):
                return f"{value.value}f"
            else:
                return str(value.value)
        
        elif value.is_temp:
            return value.name
        
        else:
            # Regular variable - needs array access for CUDA kernels
            if is_load or not (is_store):
                return f"{value.name}[i]"
            else:
                return f"{value.name}[i]"
    
    def generate_optimized_kernel(self, module: IRModule, kernel_name: str = "optimized_kernel") -> str:
        """Generate an optimized CUDA kernel with additional optimizations."""
        
        main_func = list(module.functions.values())[0]
        input_vars, output_vars, temp_vars = self._analyze_function(main_func)
        
        # Generate optimized body with additional transformations
        cuda_body = self._generate_optimized_body(main_func)
        
        # Use optimized template
        template_name = 'optimized_kernel.cu.j2' if self._template_exists('optimized_kernel.cu.j2') else 'elementwise.cu.j2'
        template = self.env.get_template(template_name)
        
        return template.render(
            kernel_name=kernel_name,
            input_vars=list(input_vars),
            output_vars=list(output_vars),
            temp_vars=list(temp_vars),
            body=cuda_body,
            optimized=True
        )
    
    def _generate_optimized_body(self, func: IRFunction) -> str:
        """Generate optimized CUDA body with loop unrolling, etc."""
        lines = []
        
        # Add thread index calculation
        lines.append("int i = blockIdx.x * blockDim.x + threadIdx.x;")
        lines.append("if (i >= n) return;")
        lines.append("")
        
        # Declare temporary variables
        temp_vars = set()
        for block in func.blocks:
            for instr in block.instructions:
                if instr.result and instr.result.is_temp:
                    temp_vars.add(instr.result.name)
        
        for temp_var in sorted(temp_vars):
            lines.append(f"float {temp_var};")
        
        if temp_vars:
            lines.append("")
        
        # Generate instructions
        for block in func.blocks:
            for instr in block.instructions:
                cuda_line = self._generate_instruction(instr)
                if cuda_line and not cuda_line.startswith('//'):
                    lines.append(cuda_line)
        
        return '\n    '.join(lines)
    
    def _template_exists(self, template_name: str) -> bool:
        """Check if a template exists."""
        try:
            self.env.get_template(template_name)
            return True
        except:
            return False

def generate_cuda_from_ir(module: IRModule, kernel_name: str = "kernel", optimized: bool = True) -> str:
    """Main entry point for IR to CUDA generation."""
    generator = IRToCUDAGenerator()
    
    if optimized:
        return generator.generate_optimized_kernel(module, kernel_name)
    else:
        return generator.generate_cuda_kernel(module, kernel_name)