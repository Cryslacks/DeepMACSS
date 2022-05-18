import json
from importlib import reload
import sys
import os
sys.path.append('language_generators/')

# Main generator for generating the code from provided semantics
def generate_code(json_str, code_gen_name, path_to_save, import_indentation=False):
    json_dict = json.loads(json_str)
    result_txt = ''
    indentation = 0
    code_gen = __import__(code_gen_name)
    reload(code_gen)
    imports, var_begin, obj_begin, obj_end = code_gen.core()
    str_code,components,variables = iterate_rows(json_dict, [], [], code_gen_name)
    
    comp_imports = code_gen.create_imports(components)
    
    if import_indentation:
        indentation, imports = fix_indentation(indentation, imports)
        indentation, comp_imports = fix_indentation(indentation, comp_imports)
    
    if ''.join(imports.split(' ')) != '':
        result_txt += imports+'\n'
    if ''.join(comp_imports.split(' ')) != '':
        result_txt += comp_imports+'\n'
    
    indentation, var_begin = fix_indentation(indentation, var_begin)
    result_txt += var_begin+'\n'

    for comp_var in variables:
        indentation, comp_var_str = fix_indentation(indentation, comp_var)
        result_txt += comp_var_str+'\n'
    result_txt += '\n'  
    
    indentation, obj_begin = fix_indentation(indentation, obj_begin)
    indentation, str_code = fix_indentation(indentation, str_code)
    indentation, obj_end = fix_indentation(indentation, obj_end)

    result_txt += obj_begin+'\n'
    result_txt += str_code+'\n'
    result_txt += obj_end
    
    f = open(path_to_save, 'w')
    f.write(result_txt)
    f.close()

# Tries to fix indentation by following basic indentation rules
def fix_indentation(curr_indentation, curr_str):
    inden_increase = ['{', '(', '>', '[']
    inden_decrease = ['}', ')', '</', ']', '/>']
    
    str_arr = curr_str.split('\n')
    for i in range(len(str_arr)):
        b_indented = False
        first_two = str_arr[i][0:2]
            
        # Check if we need to decrease indentation
        if first_two in inden_decrease:
            curr_indentation -= 1
        elif len(first_two) == 2 and first_two[0] in inden_decrease:
            curr_indentation -= 1
        else:
            if i == 0:
                str_arr[i] = curr_indentation*'  '+str_arr[i]
                b_indented = True
            else:                
                # Check if we need to increase indentation
                has_html_closing = len(str_arr[i-1].split('</')) > 1
                has_html_ending = len(str_arr[i-1].split('/>')) > 1
                last_two = str_arr[i-1][-2:]
                if last_two in inden_decrease:
                    pass
                elif has_html_closing:
                    pass
                elif has_html_ending:
                    pass
                elif last_two[1] in inden_increase:
                    curr_indentation += 1
        
        if not b_indented:   
            str_arr[i] = curr_indentation*'  '+str_arr[i]
    
        # Check if we need to decrease or decrease next indentation
        if i == len(str_arr)-1:
            first_two = ''.join(str_arr[i].split(' '))[0:2]
            if first_two in inden_decrease:
                pass
            elif first_two[0] in inden_decrease:
                pass
            else:
                last_two = str_arr[i][-2:]
                if last_two in inden_decrease:
                    pass
                elif last_two[1] in inden_increase:
                    curr_indentation += 1
        
    new_str = '\n'.join([str(e) for e in str_arr])
    return curr_indentation, new_str

# Iterates through the rows of the json dict, saving the components found to be able to have correct imports
def iterate_rows(json_dict, components, variables, code_gen_name, recurrent=False):
    str_code = ""
    exec("import "+code_gen_name)
    reference_functions = {}
    
    if "objects" in json_dict.keys():
        for i in range(len(json_dict["objects"])):
            obj = json_dict["objects"][str(i)]
            try:
                api_code, func_name = eval(code_gen_name+'.'+obj+'()')
                variables.append(api_code)
                reference_functions[obj] = func_name
            except AttributeError:
                print('[Warning] Tried to load unsupported object "'+obj+'"')
    
    for i in range(len(json_dict["rows"])):
        row = json_dict["rows"][str(i)]
        
        if 'component' in row.keys():
            comp_name = row['component']
            print('Found component:', comp_name)
            components.append(comp_name)
            
            comp_idx = components.count(comp_name)
            comp_text = ''
            comp_reference = ''
            
            if 'text' in row.keys():
                comp_text = row['text']
            if 'reference' in row.keys():
                comp_reference = row['reference']
                
            if comp_reference in reference_functions.keys():
                comp_vars,comp_code = eval(code_gen_name+'.'+comp_name+'('+str(comp_idx)+', "'+comp_text+'","'+reference_functions[comp_reference]+'", True)')
            else:
                comp_vars,comp_code = eval(code_gen_name+'.'+comp_name+'('+str(comp_idx)+', "'+comp_text+'","'+comp_reference+'")')
                
            if comp_vars != '':
                variables.append(comp_vars)
            if not recurrent:
                begin_row,end_row = eval(code_gen_name+'.row(1)')
                str_code += begin_row[0]+'\n'+comp_code+'\n'+end_row[0]+'\n'
            else:
                str_code += comp_code+'\n'
        elif 'cols' in row.keys():
            comp_codes = []
            for c in range(len(row['cols'])):
                col = row['cols'][str(c)]
                
                if 'component' in col.keys():
                    comp_name = col['component']
                    print('Found component in column:', comp_name)
                    components.append(comp_name)

                    comp_idx = components.count(comp_name)
                    comp_text = ''
                    comp_reference = ''
                    
                    if 'text' in col.keys():
                        comp_text = col['text']
                    if 'reference' in col.keys():
                        comp_reference = col['reference']
                         
                    if comp_reference in reference_functions.keys():
                        comp_vars,comp_code = eval(code_gen_name+'.'+comp_name+'('+str(comp_idx)+', "'+comp_text+'","'+reference_functions[comp_reference]+'", True)')
                    else:
                        comp_vars,comp_code = eval(code_gen_name+'.'+comp_name+'('+str(comp_idx)+', "'+comp_text+'","'+comp_reference+'")')

                    if comp_vars != '':
                        variables.append(comp_vars)
                    comp_codes.append(comp_code)
                elif 'rows' in col.keys():
                    str_code2, components2, variables2 = iterate_rows(col, components, variables, code_gen_name, True)
                    comp_codes.append(str_code2)
                    
            begin_row,end_row = eval(code_gen_name+'.row('+str(len(row['cols']))+')')
            str_code += begin_row[0]+'\n'
            for c in range(len(row['cols'])):
                str_code += begin_row[c+1]+'\n'+comp_codes[c]+'\n'+end_row[c]+"\n"
            str_code = str_code[:-1]
            str_code += end_row[len(end_row)-1]+'\n'

    return str_code[:-1], components, variables             