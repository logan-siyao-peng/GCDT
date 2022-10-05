import numpy as np
import re


def getLabelOrdered(Original_Order):
    '''
    Get the right order of lable for stacks manner.
    E.g. 
    [8,3,9,2,6,10,1,5,7,11,4] to [8,3,2,1,6,5,4,7,9,10,11]
    '''
    Original_Order = np.array(Original_Order)
    target = []
    stacks = ['root', Original_Order]
    while stacks[-1] != 'root':
        head = stacks[-1]
        if len(head) < 3:
            target.extend(head.tolist())
            del stacks[-1]
        else:
            target.append(head[0])
            temp = np.arange(len(head))
            top = head[temp[head < head[0]]]
            down = head[temp[head > head[0]]]
            del stacks[-1]
            if down.size > 0:
                stacks.append(down)
            if top.size > 0:
                stacks.append(top)

    return [x for x in target]


def get_RelationAndNucleus(label_index):

    # RelationTable = open("./data/rstdt-relations.txt", "r", encoding="utf8").read().strip().split("\n")
    RelationTable = open("./data/gum-relations.txt", "r", encoding="utf8").read().strip().split("\n")

    relation = RelationTable[label_index]
    temp = re.split(r'_', relation)
    sub1 = temp[0]
    sub2 = temp[1]

    if sub2 == 'NN':
        Nuclearity_left = 'Nucleus'
        Nuclearity_right = 'Nucleus'
        Relation_left = sub1
        Relation_right = sub1

    elif sub2 == 'NS':
        Nuclearity_left = 'Nucleus'
        Nuclearity_right = 'Satellite'
        Relation_left = 'span'
        Relation_right = sub1

    elif sub2 == 'SN':
        Nuclearity_left = 'Satellite'
        Nuclearity_right = 'Nucleus'
        Relation_left = sub1
        Relation_right = 'span'

    return Nuclearity_left, Nuclearity_right, Relation_left, Relation_right
