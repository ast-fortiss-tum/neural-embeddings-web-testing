UP_SYMBOL = "^"
DOWN_SYMBOL = "_"

#  class: Code2VecVocabs
paths_vocab = [UP_SYMBOL, DOWN_SYMBOL, 'contains']
tokens_vocab = ['a', 'p', 'button'] # => todo: text???

# target label is the title of the html page for now
def extract_target_label(soup):
    return soup.title.text

def get_immediate_parents(tag):
    parents_list = []
    for parent in tag.find_parents():
        parents_list.append(parent)
        if parent.name == 'html': break
    return parents_list

# extracts the path of 2 nodes in the html tree => eg. 'p^div^body^html'
# @param: first -> the first node to extract the path from
# @param: second -> the second node to extract the path to
# @param: soup -> soup object of the html page
def extract_path(first, second, soup):
    print(f"trying to extract path from {first.name} to {second.name}\n")
    path = []
    if first == second: return ",".join(path)
    if second in get_immediate_parents(first):
        current_node = first.parent
        path.append(f"{UP_SYMBOL}")
        while current_node != second:
            path.append(f"{current_node.name}{UP_SYMBOL}")
            current_node = current_node.parent
        return ",".join(path)
    if first in get_immediate_parents(second):
        current_node = second.parent
        path.append(f"{DOWN_SYMBOL}")
        while current_node != first:
            path.insert(0, f"{current_node.name}{DOWN_SYMBOL}")
            current_node = current_node.parent
        return ",".join(path)
    
    # Find common ancestor of both nodes
    common_ancestors = get_immediate_parents(first)
    common_ancestor = soup.html
    for ancestor in common_ancestors:
        if ancestor == second:
            common_ancestor = second
            break
        elif ancestor in second.parents:
            common_ancestor = ancestor
            break
    # Construct path from first node to common ancestor
    path.append(f"{UP_SYMBOL}")
    current_node = first
    while True:
        current_node = current_node.parent
        if current_node == common_ancestor: break
        path.append(f"{current_node.name}{UP_SYMBOL}")

    # Construct path from common ancestor to second node
    sec = []
    current_node = second
    # sec.append(f"{DOWN_SYMBOL}")
    while current_node != common_ancestor:
        current_node = current_node.parent
        sec.insert(0, f"{current_node.name}{DOWN_SYMBOL}")
    path.extend(sec)
    
    return "".join(path)

# extracts contexts only of direct neighbor nodes (parent/childs)
def extract_simple_contexts(soup):
    contexts = []
    for tag in soup.find_all():
        if tag.name in tokens_vocab:
            if len(tag.find_all()) > 0:
                for c in tag.find_all():                    
                    contexts.append(f"{tag.name},{extract_path(tag, c, soup)},{c.name}")
            elif tag.find_parent():
                contexts.append(f"{tag.name},{extract_path(tag, tag.find_parent(), soup)},{tag.find_parent().name}")
    return contexts

# final output for the row: target_label context1 context2 context3 ...
# target_label = extract_target_label(soup)
# contexts = extract_simple_contexts(soup)
# print(f'final output for the row:\n{target_label} {" ".join([con for con in contexts])}')


# todo: for each directory containing html files -> output a single text file with each row being an example 