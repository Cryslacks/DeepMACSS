def core():
	imports = "import { AppInput, AppOutput } from '@fo-navigator/app-core/lib/models';\nimport { ThemeProvider } from '@mui/material/styles';\nimport React from 'react';\nimport ReactDOM from 'react-dom';\nimport './main.css';"
	additional_code_begin = "export default async (input: AppInput): Promise<AppOutput> => {\nconst node = document.createElement('div');\nconst Root: React.FC = () => {"
	objects_begin = "return (\n<Box sx={{ p: 5, }} >"
	objects_end = "</Box>\n);\n};\nReactDOM.render(\n<ThemeProvider theme={input.theme}>\n<Root />\n</ThemeProvider>,\nnode\n);\nreturn {\nnode,\nclean: async () => {\nReactDOM.unmountComponentAtNode(node);\n},\n};\n};"
	return imports, additional_code_begin, objects_begin, objects_end

def row(nr_cols=1):
	col_begin = "<Grid item xs={"+str(12/nr_cols)+"}>"
	col_end = "</Grid>"
	row_begin = ["<Grid container  sx={{ margin: 2 }}>"]
	row_end = []
	
	for nr in range(nr_cols):
		row_begin.append(col_begin)
		row_end.append(col_end)
	
	row_end.append("\n</Grid>")
	return row_begin,row_end

def create_imports(component_list):
	component_imports = {'Header':'Box', 'Input':'TextField', 'Button':'Button', 'List':'List,ListItem', 'Datagrid':'DataGrid', 'Dropdown':'Autocomplete,TextField', 'Combobox':'Autocomplete,TextField', 'Checkbox':'Checkbox,FormGroup,FormControlLabel', 'Radiobutton':'Radio,FormGroup,FormControlLabel', 'Paragraph':'Box'}
	all_imports = ['Grid', 'Box']
	has_datagrid = False
	for comp in component_list:
		comp_imports = component_imports[comp]
		if comp == 'Datagrid':
			has_datagrid = True
			continue
		
		if len(comp_imports.split(',')) > 1:
			for item in comp_imports.split(','):
				all_imports.append(item)
		else:
			all_imports.append(comp_imports)
	
	all_imports = list(dict.fromkeys(all_imports))
	str_imports = ', '.join([str(e) for e in all_imports])
	imports = "import { "+str_imports+" } from '@mui/material';"
	
	if has_datagrid:
		imports += "\nimport { DataGrid } from '@mui/x-data-grid';"
	return imports

# Components definitions:
def Header(idx, text='', reference_text='', ref_predefined=False):
	additional_code = ""
	obj_code = "<Box sx={{ fontSize: 30, }} >"+str(text)+"</Box>"
	return additional_code, obj_code
	
def Input(idx, text='', reference_text='', ref_predefined=False):
	additional_code = "const [input"+str(idx)+", setInput"+str(idx)+"] = React.useState('');"
	if reference_text != '':
		if not ref_predefined:
			additional_code += "\nconst "+reference_text+"_func = (value) => {\nconsole.log('Input with reference "+reference_text+" changed state to:'+value);\n};"
			obj_code = "<TextField id='standard-basic' label='Input"+str(idx)+"' variant='standard' onChange={(e) => {setInput"+str(idx)+"(e.target.value);"+reference_text+"_func(e.target.value)}}/>"
		else:
			obj_code = "<TextField id='standard-basic' label='Input"+str(idx)+"' variant='standard' onChange={(e) => {setInput"+str(idx)+"(e.target.value);"+reference_text+"(e.target.value)}}/>"
		return additional_code, obj_code

	obj_code = "<TextField id='standard-basic' label='Input"+str(idx)+"' variant='standard' onChange={(e) => {setInput"+str(idx)+"(e.target.value)}}/>"
	return additional_code, obj_code
	
def Button(idx, text='', reference_text='', ref_predefined=False):
	additional_code = ""

	if reference_text != '':
		if not ref_predefined:
			additional_code = "const "+reference_text+"_func = () => {\nconsole.log('Button with reference "+reference_text+" clicked');\n};"
			obj_code = "<Button variant='contained' onClick={"+reference_text+"_func}>"+str(text)+"</Button>"
		else:
			obj_code = "<Button variant='contained' onClick={"+reference_text+"}>"+str(text)+"</Button>"
			
		return additional_code, obj_code
	
	obj_code = "<Button variant='contained' onClick={() => alert('Clicked button "+str(text)+"')}>"+str(text)+"</Button>"
	return additional_code, obj_code
	
def List(idx, text='', reference_text='', ref_predefined=False):
	additional_code = ""

	if reference_text != '':
		if not ref_predefined:
			additional_code = "const "+reference_text+"_func = () => {\nreturn ['"+reference_text+"1', '"+reference_text+"2', '"+reference_text+"3'];\n}"
			obj_code = "<List>\n{"+reference_text+"_func().map(el => (<ListItem>{el}</ListItem>))}\n</List>"
		else:
			obj_code = "<List>\n{"+reference_text+"().map(el => (<ListItem>{el}</ListItem>))}\n</List>"

		return additional_code, obj_code
		
	obj_code = "<List>\n<ListItem>Item1</ListItem>\n<ListItem>Item2</ListItem>\n<ListItem>Item3</ListItem>\n</List>"
	return additional_code, obj_code
	
def Datagrid(idx, text='', reference_text='', ref_predefined=False):
	if reference_text != '':
		if not ref_predefined:
			additional_code = "const "+reference_text+"_func = () => {\nreturn {\n'columns': [{field: 'col1', headerName:'"+reference_text+" 1', width:150}, {field: 'col2', headerName:'"+reference_text+" 2', width:150}],\n'rows': [{id: 1, col1:'"+reference_text+"1_row1', col2:'"+reference_text+"2_row1'}, {id: 2, col1:'"+reference_text+"1_row2', col2:'"+reference_text+"2_row2'}]\n}\n}"
			obj_code = "<DataGrid \n  style={{ height: 300, width: '100%' }}\n  rows={"+reference_text+"_func().rows}\n  columns={"+reference_text+"_func().columns} />"
		else:
			additional_code = ""
			obj_code = "<DataGrid \n  style={{ height: 300, width: '100%' }}\n  rows={"+reference_text+"().rows}\n  columns={"+reference_text+"().columns} />"
		
		return additional_code, obj_code

	additional_code = "const columns"+str(idx)+" = [\n{field: 'col1', headerName:'Column 1', width:150},\n{field: 'col2', headerName:'Column 2', width:150}\n];\n"
	additional_code += "const rows"+str(idx)+" = [\n{id: 1, col1:'column1_row1', col2:'column2_row1'},\n{id: 2, col1:'column1_row2', col2:'column2_row2'},\n{id: 3, col1:'column1_row3', col2:'column2_row3'}\n];"
	obj_code = "<DataGrid \n  style={{ height: 300, width: '100%' }}\n  rows={rows"+str(idx)+"}\n  columns={columns"+str(idx)+"} />"
	return additional_code, obj_code
	
def Dropdown(idx, text='', reference_text='', ref_predefined=False):
	obj_code = "<Autocomplete\n"
	additional_code = ""

	if reference_text != '':
		if not ref_predefined:
			obj_code += "  options={"+reference_text+"_func()}\n"
			additional_code = "const "+reference_text+"_func = () => {\nreturn [\n{label:'"+reference_text+"1', value:1},\n{label:'"+reference_text+"2', value:2},\n{label:'"+reference_text+"3', value:3}\n];\n}\n"
		else:
			obj_code += "  options={"+reference_text+"()}\n"
	else:
		obj_code += "  options={dropdown_options"+str(idx)+"}\n"
		additional_code = "const dropdown_options"+str(idx)+" = [\n{label:'First option', value:1},\n{label:'Second option', value:2},\n{label:'Third option', value:3}\n];\n"

	additional_code += "const [dropdown"+str(idx)+", setDropdown"+str(idx)+"] = React.useState('');"
	
	obj_code += "  renderInput={(params) => <TextField {...params} label='Dropdown_"+str(idx)+"' />}\n"
	obj_code += "  onChange={(_,v) => setDropdown"+str(idx)+"(v.value)} />"
	return additional_code, obj_code
	
def Combobox(idx, text='', reference_text='', ref_predefined=False):
	obj_code = "<Autocomplete\n"
	additional_code = ""
	
	if reference_text != '':
		if not ref_predefined:
			obj_code += "  options={"+reference_text+"_func()}\n"
			additional_code = "const "+reference_text+"_func = () => {\nreturn [\n{label:'"+reference_text+"1', value:1},\n{label:'"+reference_text+"2', value:2},\n{label:'"+reference_text+"3', value:3}\n];\n}\n"
		else:
			obj_code += "  options={"+reference_text+"()}\n"
	else:
		obj_code += "  options={combobox_options"+str(idx)+"}\n"
		additional_code = "const combobox_options"+str(idx)+" = [\n{label:'First option', value:1},\n{label:'Second option', value:2},\n{label:'Third option', value:3}\n];\n"

	additional_code += "const [combobox"+str(idx)+", setCombobox"+str(idx)+"] = React.useState('');"
	
	obj_code += "  renderInput={(params) => <TextField {...params} label='Combobox_"+str(idx)+"' />}\n"
	obj_code += "  onChange={(_,v) => setCombobox"+str(idx)+"(v.value)} />"
	return additional_code, obj_code
	
def Checkbox(idx, text='', reference_text='', ref_predefined=False):
	additional_code = ""
	obj_code = "<FormGroup>\n"
	obj_code += "<FormControlLabel control={<Checkbox />} label='"+str(text)+"' />\n"
	obj_code += "</FormGroup>"
	return additional_code, obj_code
	
def Radiobutton(idx, text='', reference_text='', ref_predefined=False):
	additional_code = ""
	obj_code = "<FormGroup>\n"
	obj_code += "<FormControlLabel control={<Radio />} label='"+str(text)+"' />\n"
	obj_code += "</FormGroup>"
	return additional_code, obj_code
	
def Paragraph(idx, text='', reference_text='', ref_predefined=False):
	additional_code = ""
	obj_code = "<Box sx={{ fontSize: 14, }} >"+str(text)+"</Box>"
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
	api_code = "const test_api = () => {\nconsole.log('test');\n}"
	func_name = "test_api"
	return api_code, func_name