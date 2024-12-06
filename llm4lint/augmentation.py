"""augmentation module generates more data by modifying syntax tree"""
import ast

example_code = """
l = [1,2,3,4]
for i in l:
    l.append(1)
"""


class FindNames(ast.NodeVisitor):
    """finds all unique name ids: 
    call FindNamesobj.visit(tree) to store names in FindNamesobj.node_ids:set atrr"""
    def __init__(self) -> None:
        super().__init__()
        self.node_ids = set()

    def visit_Name(self, node):
        #print(node.id)
        self.node_ids.add(node.id)
        return self.node_ids


class NameNodeTransformer(ast.NodeTransformer):
    """Transforms all instances of old_name node to new_name node"""
    def __init__(self, old_name: ast.Name, new_name: ast.Name) -> None:
        super().__init__()
        self.old_name = old_name
        self.new_name = new_name

    def visit_Name(self, node):
        if node.id == self.old_name.id:
            # print(node.id, "changed to", self.new_name.id)
            return ast.Name(id=self.new_name.id, ctx=node.ctx)
        return node
    
tree = ast.parse(example_code)
name_l = ast.Name(id='l')
new_name = ast.Name(id='test')
new_tree = ast.fix_missing_locations(NameNodeTransformer(old_name=name_l, new_name=new_name).visit(tree))
print(ast.unparse(new_tree))

# tree = ast.parse(example_code)
# var_name_visitor = FindNames()
# print(var_name_visitor.visit(tree))
# print(var_name_visitor.var_nodes)