"""augmentation module generates more data by modifying syntax tree"""
import ast
from typing import List, Set
from copy import copy
from random import shuffle

example_code = """
l = [1,2,3,4]
for i in l:
    l.append(1)
"""

names = ['to_do', 'find', 'data_for', 'variable_names']

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


def random_name():
    raise NotImplementedError

class NameRandomizer():
    """handles accessing names, names list is unique. so it ensures same name is not returned again"""
    def __init__(self, names: List[str]) -> None:
        self.names = copy(names)
        shuffle(self.names)
        self.index = 0
    
    def __len__(self) -> int:
        return len(self.names)
    
    def pop(self) -> ast.Name:
        """modifies the self.names list"""
        return ast.Name(id=self.names.pop())
    
    def get_name(self) -> ast.Name:
        """it does not modify self.names list"""
        name = self.names[self.index]
        self.index += 1
        return ast.Name(id=name)


def augment_code_names(src_code: str) -> str:
    """Returns augmented src code by transforming variable names"""
    tree = ast.parse(src_code)
    # find all name node ids
    var_name_finder = FindNames()
    var_name_finder.visit(tree)
    var_names: Set[str] = var_name_finder.node_ids
    name_randomizer = NameRandomizer(names)
    # change each name node to new node
    for name in var_names:
        new_name = name_randomizer.pop()
        tree = ast.fix_missing_locations(NameNodeTransformer(old_name=ast.Name(id=name), new_name=new_name).visit(tree))
    augmented_code = ast.unparse(tree)
    return augmented_code


# tree = ast.parse(example_code)
# name_l = ast.Name(id='l')
# new_name = ast.Name(id='test')
# new_tree = ast.fix_missing_locations(NameNodeTransformer(old_name=name_l, new_name=new_name).visit(tree))
# print(ast.unparse(new_tree))

print(augment_code_names(example_code))