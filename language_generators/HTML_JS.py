def core():
	imports = "<html>\n<head>"
	additional_code_begin = "<script>"
	objects_begin = "</script>\n</head>\n<body>"
	objects_end = "</body>\n</html>"
	return imports, additional_code_begin, objects_begin, objects_end

def row(nr_cols=1):
	col_begin = "<div class='column'>"
	col_end = "</div>"
	row_begin = ["<div class='row'>"]
	row_end = []
	
	for nr in range(nr_cols):
		row_begin.append(col_begin)
		row_end.append(col_end)
	
	row_end.append("\n</div>")
	return row_begin,row_end

def create_imports(component_list):
	# This could be added to support basic styling imports such as bootstrap and so on for different components
	# Im gonna apply this to create a basic formating via custom css
	styling_begin = '<style>\n'
	styling_end = '</style>'
	
	custom_css =  '.row {\ndisplay: flex;\nborder: 1px dotted green;\n}\n'
	custom_css += '.column {\nmargin: 10px;\nborder: 1px dotted red;\n}\n'
	return styling_begin + custom_css + styling_end

# Components definitions:
def Header(idx, text='', reference_text='', ref_predefined=False):
	additional_code = ""
	obj_code = "<h2 class='header'>"+str(text)+"</h2>"
	return additional_code, obj_code
	
def Input(idx, text='', reference_text='', ref_predefined=False):
	additional_code = ""
	label_text = "Input"+str(idx)
	obj_code = "<div class='label-item'>\n"
	
	if text != '':
		label_text = text
	if reference_text != '':
		if not ref_predefined:
			additional_code = "function "+reference_text+"_func() {\nvar value = document.getElementById('Input"+str(idx)+"').value;\nconsole.log('Input with reference "+reference_text+" changed state to:'+value);\n}"
			obj_code += "<Input id='Input"+str(idx)+"' onchange='"+reference_text+"_func()'></input>\n"
		else:
			obj_code += "<Input id='Input"+str(idx)+"' onchange='"+reference_text+"()'></input>\n"			
		obj_code += "<label for='Input"+str(idx)+"'>\n"+label_text+"\n</label>\n"
		obj_code += "</div>"
		return additional_code, obj_code

	obj_code += "<Input id='Input"+str(idx)+"'></input>\n"
	obj_code += "<label for='Input"+str(idx)+"'>\n"+label_text+"\n</label>\n"
	obj_code += "</div>"
	return additional_code, obj_code
	
def Button(idx, text='', reference_text='', ref_predefined=False):
	additional_code = ""

	if reference_text != '':
		if not ref_predefined:
			additional_code = "function "+reference_text+"_func() {\nconsole.log('Button with reference "+reference_text+" clicked');\n}"
			obj_code = "<Button id='Button"+str(idx)+"' onclick='"+reference_text+"_func()'>\n"+text+"\n</Button>"
		else:
			obj_code = "<Button id='Button"+str(idx)+"' onclick='"+reference_text+"()'>\n"+text+"\n</Button>"			
		return additional_code, obj_code
	
	obj_code = "<Button id='Button"+str(idx)+"'>\n"+text+"\n</Button>"
	return additional_code, obj_code
	
def List(idx, text='', reference_text='', ref_predefined=False):
	additional_code = ""

	if reference_text != '':
		if not ref_predefined:
			additional_code = "function "+reference_text+"_func() {\nvar list = document.getElementById('List"+str(idx)+"');\nvar items = ['Item1', 'Item2', 'Item3'];\n\nfor (const item of items) {\nvar li = document.createElement('li');\nli.textContent = item;\nlist.appendChild(li);\n}\n}\n"
			additional_code += "window.addEventListener('load', function () {\n"+reference_text+"_func();\n});"
		else:
			additional_code = "window.addEventListener('load', function () {\n"+reference_text+"();\n});"			
		obj_code = "<ul id='List"+str(idx)+"'></ul>"
		return additional_code, obj_code
		
	obj_code = "<ul>\n<li>Item1</li>\n<li>Item2</li>\n<li>Item3</li>\n</ul>"
	return additional_code, obj_code
	
def Datagrid(idx, text='', reference_text='', ref_predefined=False):
	return List(idx, text, reference_text)
	
def Dropdown(idx, text='', reference_text='', ref_predefined=False):
	return Paragraph(idx, 'Dropdown', reference_text)
	obj_code = "<Autocomplete\n"

	if reference_text != '':
		obj_code += "  options={"+reference_text+"_func()}\n"
		additional_code = "const "+reference_text+"_func = () => {\nreturn [\n{label:'"+reference_text+"1', value:1},\n{label:'"+reference_text+"2', value:2},\n{label:'"+reference_text+"3', value:3}\n];\n}"
	else:
		obj_code += "  options={dropdown_options"+str(idx)+"}\n"
		additional_code = "const dropdown_options"+str(idx)+" = [\n{label:'First option', value:1},\n{label:'Second option', value:2},\n{label:'Third option', value:3}\n];\n"

	additional_code += "const [dropdown"+str(idx)+", setDropdown"+str(idx)+"] = React.useState('');"
	
	obj_code += "  renderInput={(params) => <TextField {...params} label='Dropdown_"+str(idx)+"' />}\n"
	obj_code += "  onChange={(_,v) => setDropdown"+str(idx)+"(v.value)} />"
	return additional_code, obj_code
	
def Combobox(idx, text='', reference_text='', ref_predefined=False):
	return Paragraph(idx, 'Combobox', reference_text)
	obj_code = "<Autocomplete\n"
	
	if reference_text != '':
		obj_code += "  options={"+reference_text+"_func()}\n"
		additional_code = "const "+reference_text+"_func = () => {\nreturn [\n{label:'"+reference_text+"1', value:1},\n{label:'"+reference_text+"2', value:2},\n{label:'"+reference_text+"3', value:3}\n];\n}"
	else:
		obj_code += "  options={combobox_options"+str(idx)+"}\n"
		additional_code = "const combobox_options"+str(idx)+" = [\n{label:'First option', value:1},\n{label:'Second option', value:2},\n{label:'Third option', value:3}\n];\n"

	additional_code += "const [combobox"+str(idx)+", setCombobox"+str(idx)+"] = React.useState('');"
	
	obj_code += "  renderInput={(params) => <TextField {...params} label='Combobox_"+str(idx)+"' />}\n"
	obj_code += "  onChange={(_,v) => setCombobox"+str(idx)+"(v.value)} />"
	return additional_code, obj_code
	
def Checkbox(idx, text='', reference_text='', ref_predefined=False):
	additional_code = ""
	label_text = "Checkbox"+str(idx)
	obj_code = "<div class='label-item'>\n"
	if text != '':
		label_text = text
	if reference_text != '':
		if not ref_predefined:
			additional_code = "function "+reference_text+"_func() {\nconsole.log('Checkbox with reference "+reference_text+" clicked');\n}"
			obj_code += "<Input id='Checkbox"+str(idx)+"' onchange='"+reference_text+"_func()' type='checkbox'></input>\n"
		else:
			obj_code += "<Input id='Checkbox"+str(idx)+"' onchange='"+reference_text+"()' type='checkbox'></input>\n"			
		obj_code += "<label for='Checkbox"+str(idx)+"'>\n"+label_text+"\n</label>\n"
		obj_code += "</div>"
		return additional_code, obj_code

	obj_code += "<Input id='Checkbox"+str(idx)+"' type='checkbox'></input>\n"
	obj_code += "<label for='Checkbox"+str(idx)+"'>\n"+label_text+"\n</label>\n"
	obj_code += "</div>"
	return additional_code, obj_code
	
def Radiobutton(idx, text='', reference_text='', ref_predefined=False):
	additional_code = ""
	label_text = "Radiobutton"+str(idx)
	obj_code = "<div class='label-item'>\n"
	
	if text != '':
		label_text = text
	if reference_text != '':
		if not ref_predefined:
			additional_code = "function "+reference_text+"_func() {\nconsole.log('Radiobutton with reference "+reference_text+" clicked');\n}"
			obj_code += "<Input id='Radiobutton"+str(idx)+"' onchange='"+reference_text+"_func()' type='radio'></input>\n"
		else:
			obj_code += "<Input id='Radiobutton"+str(idx)+"' onchange='"+reference_text+"()' type='radio'></input>\n"
		obj_code += "<label for='Radiobutton"+str(idx)+"'>\n"+label_text+"\n</label>\n"
		obj_code += "</div>"
		return additional_code, obj_code

	obj_code += "<Input id='Radiobutton"+str(idx)+"' type='radio'></input>\n"
	obj_code += "<label for='Radiobutton"+str(idx)+"'>\n"+label_text+"\n</label>\n"
	obj_code += "</div>"
	return additional_code, obj_code
	
def Paragraph(idx, text='', reference_text='', ref_predefined=False):
	additional_code = ""
	obj_code = "<p>"+str(text)+"</p>"
	return additional_code, obj_code

# Specified API objects:
def Login():
	api_code = ""
	func_name = ""
	return api_code, func_name
	
def Stores():
	api_code = ""
	func_name = ""
	return api_code, func_name
	
def Tennants():
	api_code = ""
	func_name = ""
	return api_code, func_name
	
def Test():
	api_code = "function test_api() {\nconsole.log('test');\n}"
	func_name = "test_api"
	return api_code, func_name