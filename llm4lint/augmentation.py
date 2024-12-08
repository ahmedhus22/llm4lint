"""augmentation module generates more data by modifying syntax tree"""
import ast
from typing import List, Set
from copy import copy
from random import shuffle
from pathlib import Path
from random import randrange

from datasets import load_dataset, Dataset

from augdata import names, NAME_TRANSFORM_EXCEPTIONS

example_code = """
l = [1,2,3,4]
for i in l:
    l.append(1)
"""

pos_dataset = load_dataset("csv", data_files=str(Path("../datasets/clean_code_aug.csv")))["train"]

class FindNames(ast.NodeVisitor):
    """finds all unique name ids: 
    call FindNamesobj.visit(tree) to store names in FindNamesobj.node_ids:set atrr"""
    def __init__(self) -> None:
        super().__init__()
        self.node_ids = set()

    def visit_Name(self, node):
        #print(node.id)
        if node.id in NAME_TRANSFORM_EXCEPTIONS:
            return None
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
    if src_code is None: return src_code
    tree = ast.parse(src_code)
    # find all name node ids
    var_name_finder = FindNames()
    var_name_finder.visit(tree)
    var_names: Set[str] = var_name_finder.node_ids
    name_randomizer = NameRandomizer(names)
    # change each name node to new node
    for name in var_names:
        if name in NAME_TRANSFORM_EXCEPTIONS:
            continue
        new_name = name_randomizer.pop()
        tree = ast.fix_missing_locations(NameNodeTransformer(old_name=ast.Name(id=name), new_name=new_name).visit(tree))
    augmented_code = ast.unparse(tree)
    return augmented_code

def _addcodelines(code: str) -> str:
    code_with_lnos = ""
    code_lines = code.split("\n")
    for index, line in enumerate(code_lines):
        code_with_lnos += str(index+1) + "   " + line + "\n"
    return code_with_lnos

def augment_data(examples):
    """augments data for input column in a batch"""
    inputs = []
    outputs = []
    # disable positional augmentations for now (need to find a dataset)
    AUG_POSITIONS = 15
    NO_POSITIONAL_CHANGE = 30
    for code, label, lineno in zip(examples["input"], examples["output"], examples["lineno"]):
        augmented_sequences = []
        for _ in range(NO_POSITIONAL_CHANGE):
            augmented_sequences.append(_addcodelines(augment_code_names(code)))
        # linenos need to be updated as well
        positional_labels = []
        for i in range(AUG_POSITIONS):
            aug_code: str = augment_code_names(code)
            pos_dataset.shuffle(seed=i*i, keep_in_memory=True)
            aug_prefix: str = pos_dataset[i]["input"]
            aug_postfix: str = pos_dataset[i+1]["input"]
            no_prefix_lines = aug_prefix.count("\n")
            # additional 1 is added to line number because of new line before code
            positional_labels.append(str(no_prefix_lines+lineno+1)+" - "+label)
            aug_code = aug_prefix + "\n" + aug_code + "\n" + aug_postfix
            aug_code = _addcodelines(aug_code)
            augmented_sequences.append(aug_code)
        inputs += [code] + augmented_sequences
        outputs += [str(lineno)+" - "+label] + [str(lineno)+" - "+label] * (NO_POSITIONAL_CHANGE)  + positional_labels
    return {"input": inputs, "output": outputs}

def augment_dataset(examples: Path, save_path: Path, save_format: str="csv") -> Dataset:
    """create new data points for given examples"""
    # original_data: pd.DataFrame = pd.read_csv(examples)[0]
    dataset = load_dataset("csv", data_files=str(examples))["train"]
    aug_dataset = dataset.map(augment_data, batched=True, remove_columns=dataset.column_names, batch_size=100)
    if save_format == "csv":
        aug_dataset.to_csv(save_path)
    else:
        aug_dataset.save_to_disk(save_path)
    return aug_dataset


def main():
    augment_dataset(
        Path("../datasets/examples.csv"),
        save_path=Path("../datasets/examples_aug.csv"),
        save_format="csv"
    )

if __name__=="__main__":
    main()