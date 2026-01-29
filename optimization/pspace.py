#!/usr/bin/env python3
import re
import sys
from ConfigSpace import ConfigurationSpace, Integer, Float, Categorical
from ConfigSpace.conditions import EqualsCondition

# Helper to safely find matching brace
def find_matching_brace(s, start_idx):
    balance = 0
    in_quote = False
    quote_char = None
    for i in range(start_idx, len(s)):
        char = s[i]
        if in_quote:
            if char == quote_char:
                # Handle escaped quotes? Assuming simple for now
                if i > 0 and s[i-1] == '\\':
                    pass
                else:
                    in_quote = False
        else:
            if char == '"' or char == "'":
                in_quote = True
                quote_char = char
            elif char == '{':
                balance += 1
            elif char == '}':
                balance -= 1
                if balance == 0:
                    return i
    return -1

def get_param_name(prefix_text, param_counter):
    base_name_match = re.search(r'(\w+)\s*=?\s*$', prefix_text)
    param_base_name = base_name_match.group(1) if base_name_match else "param"
    param_counter[param_base_name] = param_counter.get(param_base_name, -1) + 1
    return f"{param_base_name}_{param_counter[param_base_name]}"

def parse_block_recursive(text, config_space, param_counter):
    """
    Parses a text block, identifying static text and parameters (including nested optionals).
    Returns a list of token dicts.
    """
    tokens = []
    i = 0
    length = len(text)
    
    while i < length:
        # Find start of next parameter block
        start = text.find('{', i)
        
        # If no more blocks, add remaining text and finish
        if start == -1:
            if i < length:
                tokens.append({'type': 'text', 'content': text[i:]})
            break
            
        # Add text before the block
        if start > i:
             tokens.append({'type': 'text', 'content': text[i:start]})
             
        # Find matching close brace
        end = find_matching_brace(text, start)
        if end == -1:
            raise ValueError(f"Unbalanced braces starting at position {start} in: ...{text[start:min(start+20, len(text))]}...")
            
        block_full = text[start:end+1]         # e.g. {optional NAME ...}
        block_content = text[start+1:end]      # e.g. optional NAME ...
        
        # Parse the block content to determine type
        # Tokenize simply by spaces to get the tag, respecting quotes is not strictly needed for the tag
        # but we need to handle {list "..." ...}
        
        parts = block_content.split(maxsplit=1)
        tag = parts[0] if parts else ""
        
        token_info = {}
        
        if tag == "optional":
            if len(parts) < 2:
                raise ValueError(f"Invalid optional block, missing name: {block_full}")
            
            rest = parts[1].strip()
            # The FIRST word of rest is the NAME. The rest is the body.
            # But the body might start immediately or after space.
            # e.g. "NAME {..." or "NAME text {..."
            
            name_match = re.match(r'^(\w+)\s*(.*)', rest, re.DOTALL)
            if not name_match:
                 raise ValueError(f"Could not parse optional name in: {block_full}")
            
            opt_name = name_match.group(1)
            body_text = name_match.group(2)
            
            # Create the boolean switch parameter
            # We use Categorical for conditions support
            hp = Categorical(opt_name, ["false", "true"], default="false")
            if opt_name not in config_space.get_hyperparameter_names():
                config_space.add_hyperparameter(hp)
            
            # Recursively parse the body
            # We need to maintain a separate param counter? 
            # Or share the global one? Sharing global one is better for uniqueness.
            children = parse_block_recursive(body_text, config_space, param_counter)
            
            # Apply conditions to all parameters found in children
            def get_all_param_names(toks):
                names = []
                for t in toks:
                    if t['type'] == 'param':
                        names.append(t['param_name'])
                    elif t['type'] == 'optional':
                        # The optional block itself is a container, its internal boolean is 'param_name'
                        # We only condition the switch parameter of the nested block.
                        # The nested block's children are already conditioned by the nested switch.
                        names.append(t['param_name'])
                return names

            child_params = get_all_param_names(children)
            
            parent_hp = config_space.get_hyperparameter(opt_name)
            for child_name in child_params:
                child_hp = config_space.get_hyperparameter(child_name)
                # Avoid adding duplicate conditions if already exists (ConfigSpace might error or ignore)
                # But ConfigSpace doesn't easily let us check existence of specific condition.
                # Assuming tree structure, we process top-down.
                cond = EqualsCondition(child_hp, parent_hp, "true")
                config_space.add_condition(cond)
            
            token_info = {
                'type': 'optional',
                'param_name': opt_name,
                'children': children
            }
            
        elif tag == "list":
            # format: {list "val1" "val2"}
            # Extract quoted strings
            list_values = re.findall(r'"([^"]*)"', block_content)
            if not list_values:
                 raise ValueError(f"List parameter must contain at least one quoted value: {block_full}")
            
            # Name deduction based on usage: 
            # In recursive context, we need to know the text preceding *this block* in the *immediate parent text*.
            # But here `text` is the local string being parsed.
            # `scan_text` stores text in `tokens`.
            # We can reconstruct prefix from the last text token?
            prefix = ""
            if tokens and tokens[-1]['type'] == 'text':
                prefix = tokens[-1]['content']
            
            param_name = get_param_name(prefix, param_counter)
            
            hp = Integer(param_name, (0, len(list_values) - 1))
            config_space.add_hyperparameter(hp)
            
            token_info = {
                'type': 'param',
                'param_name': param_name,
                'param_type': 'list',
                'list_values': list_values
            }
            
        elif tag in ['exp2', 'exp2_or_0', 'int', 'float', 'bool']:
            # format: {type lower upper} or {bool}
            args = block_content.split()
            # args[0] is tag
            
            prefix = ""
            if tokens and tokens[-1]['type'] == 'text':
                prefix = tokens[-1]['content']
            param_name = get_param_name(prefix, param_counter)
            
            if tag == 'exp2_or_0':
                if len(args) < 3: raise ValueError(f"exp2_or_0 requires lower and upper bounds: {block_full}")
                lower, upper = int(args[1]), int(args[2])
                range_size = upper - lower + 2
                hp = Integer(param_name, (0, range_size - 1))
                token_info.update({'lower': lower, 'upper': upper})
                
            elif tag == 'exp2':
                if len(args) < 3: raise ValueError(f"exp2 requires lower and upper bounds: {block_full}")
                lower, upper = int(args[1]), int(args[2])
                hp = Integer(param_name, (lower, upper))
                
            elif tag == 'int':
                if len(args) < 3: raise ValueError(f"int requires lower and upper bounds: {block_full}")
                lower, upper = int(args[1]), int(args[2])
                hp = Integer(param_name, (lower, upper))
                
            elif tag == 'float':
                if len(args) < 3: raise ValueError(f"float requires lower and upper bounds: {block_full}")
                lower, upper = float(args[1]), float(args[2])
                hp = Float(param_name, (lower, upper))
                
            elif tag == 'bool':
                lower = 0
                upper = 1
                if len(args) >= 2: lower = int(args[1])
                if len(args) >= 3: upper = int(args[2])
                hp = Integer(param_name, (lower, upper)) # Using Integer for compat with existing logic
            
            config_space.add_hyperparameter(hp)
            token_info.update({
                'type': 'param',
                'param_name': param_name,
                'param_type': tag
            })
            
        else:
            # Unknown tag - fail explicitly
            raise ValueError(f"Unknown parameter block tag '{tag}' in: {block_full}")
            
        tokens.append(token_info)
        i = end + 1

    return tokens

def parse_pspace(pspace_path: str):
    config_space = ConfigurationSpace()
    lines_template = []
    param_counter = {}
    
    with open(pspace_path, 'r') as f:
        for line in f:
            original_line = line.rstrip('\n')
            stripped = line.strip()
            
            if not stripped or stripped.startswith('#') or stripped.startswith('['):
                lines_template.append({'type': 'static', 'content': original_line})
                continue
                
            # Parse line recursively
            try:
                line_tokens = parse_block_recursive(original_line, config_space, param_counter)
                lines_template.append({'type': 'templated_line', 'tokens': line_tokens})
            except Exception as e:
                print(f"Error parsing line: {line}", file=sys.stderr)
                print(f"Error: {e}", file=sys.stderr)
                raise  # Fail explicitly instead of masking the error
    
    return lines_template, config_space

def generate_output_recursive(tokens, config):
    out = ""
    for token in tokens:
        if token['type'] == 'text':
            out += token['content']
            
        elif token['type'] == 'optional':
            # Check the switch
            name = token['param_name']
            val = config.get(name)
            
            # val should be "true" or "false" (string) if Categorical
            if val == "true":
                # Render content
                out += generate_output_recursive(token['children'], config)
            else:
                # Disabled -> Render nothing
                pass
                
        elif token['type'] == 'param':
            name = token['param_name']
            val = config.get(name)
            p_type = token['param_type']
            
            if val is None:
                # Parameter value is None - this should not happen for active parameters
                raise ValueError(f"Parameter '{name}' has None value in configuration - this indicates a bug in the config generation")

            if p_type == 'list':
                list_values = token['list_values']
                idx = int(val)
                out += list_values[idx]
                
            elif p_type == 'exp2':
                out += str(2 ** int(val))
                
            elif p_type == 'exp2_or_0':
                int_val = int(val)
                out += "0" if int_val == 0 else str(2 ** (token.get('lower', 0) + int_val - 1))
                
            elif p_type == 'int':
                out += str(int(val))
                
            elif p_type == 'float':
                out += str(float(val))
                
            elif p_type == 'bool':
                out += 'true' if int(val) != 0 else 'false'
                
            else:
                out += str(val)
                
    return out

def config_to_file(lines_template: list, config: dict, output_path: str):
    with open(output_path, 'w') as f:
        for line_info in lines_template:
            if line_info['type'] == 'static':
                f.write(line_info['content'] + '\n')
            elif line_info['type'] == 'templated_line':
                # Generate
                # If the line becomes empty because of optional status, do we print newline?
                # The user probably expects the line to vanish if features are purely optional.
                # But if pure text remains?
                # Let's print what we generate.
                
                generated = generate_output_recursive(line_info['tokens'], config)
                
                # Heuristic: if generated line is empty or just whitespace?
                if generated.strip():
                    f.write(generated + '\n')

