from bs4 import BeautifulSoup
UP_SYMBOL = "^"
DOWN_SYMBOL = "_"

#  class: Code2VecVocabs
paths_vocab = [UP_SYMBOL, DOWN_SYMBOL, 'contains']
tokens_vocab = ['p', 'a', 'button', 'form', 'img', 'section', 'h1', 'h2', 'h3', 'ul', 'ol', 'li', 'div'] # => todo: text???, 3 categories: textual content, interactive elements, structural elements
structural_tags = ['head', 'body', 'section', 'ul', 'ol']

# @param: soup -> soup object of the html page
# returns list the nodes of possible target labels (names of semantically important tags)
def extract_target_labels(soup):
    target_labels = []
    for tag in soup.find_all():
        if tag.name == 'div' and len(tag.find_all(recursive=False)) > 1:
            target_labels.append(tag)
        elif tag.name in structural_tags:
            target_labels.append(tag)
    return target_labels

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
    # print(f"trying to extract path from {first.name} to {second.name}\n")
    path = []
    if first == second: return ",".join(path)
    if second in get_immediate_parents(first):
        current_node = first.parent
        path.append(f"{UP_SYMBOL}")
        while current_node != second:
            path.append(f"{current_node.name}{UP_SYMBOL}")
            current_node = current_node.parent
        return "".join(path)
    if first in get_immediate_parents(second):
        current_node = second.parent
        while current_node != first:
            path.insert(0, f"{current_node.name}{DOWN_SYMBOL}")
            current_node = current_node.parent
        path.insert(0, f"{DOWN_SYMBOL}")
        return "".join(path)
    
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
    while current_node != common_ancestor:
        current_node = current_node.parent
        sec.insert(0, f"{current_node.name}{DOWN_SYMBOL}")
    path.extend(sec)
    
    return "".join(path)

# extracts contexts of tags from vocabulary
# @param: node -> html tag of which to extract contexts from 
# @param: simple, default=False -> only extract contexts of direct neighbor nodes (parent/childs) of tags from vocabulary (=> path lenght = 1)
# returns array of strings of the form: 'tag1,path,tag2'
def extract_contexts(node, simple=False):
    contexts = []
    for tag in node.find_all():
        if tag.name in tokens_vocab:
            if len(tag.find_all()) > 0: # ensure that node is not leaf
                for c in tag.find_all(recursive=not simple):                    
                    contexts.append(f"{tag.name},{extract_path(tag, c, node)},{c.name}")
            elif tag.find_parent():
                contexts.append(f"{tag.name},{extract_path(tag, tag.find_parent(), node)},{tag.find_parent().name}")
    return contexts

# prints rows as defined in the code2vec docs, -> final output for a row: 'target_label context1 context2 context3 ...'
# @param: soup -> soup object of the html page
def print_examples(soup):
    target_labels = extract_target_labels(soup)
    for target_label in target_labels:
        contexts = extract_contexts(target_label)
        print(f'{target_label.name}|{str(target_label.attrs).replace(" ", "")} {" ".join([con for con in contexts])}\n')    
    print(f"***Printed {len(target_labels)} examples for {soup.title.string}***") # TODO remove
    
print_examples(BeautifulSoup(open('code2vec/testHTML/MDN_webdocs.html'), 'html.parser'))

# soup = BeautifulSoup(open('code2vec/testHTML/MDN_webdocs.html'), 'html.parser')
# tl = extract_target_labels(soup)
# div = tl[-2]
# print(extract_path(div.a.title, soup.body, soup))

# c = """
#         <html>
#         <head>
#             <title>Sample Page</title>
#         </head>
#         <body>
#             <div>
#                 <p>Paragraph inside div</p>
#                 <div>
#                     <p>Paragraph inside div inside div</p>
#                 </div>
#             </div>
#             <button>
#                 <p>Paragraph inside button</p>
#             </button>
#         </body>
#         </html>
#         """
# soup = BeautifulSoup(c, "html.parser")
# print_examples(soup)

# todo: for each directory containing html files -> output a single text file with each row being an example, call print_example method