from bs4 import BeautifulSoup
import re
from argparse import ArgumentParser
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

UP_SYMBOL = "^"
DOWN_SYMBOL = "_"

#  class: Code2VecVocabs
paths_vocab = [UP_SYMBOL, DOWN_SYMBOL, 'contains']
tokens_vocab = ['p', 'a', 'button', 'form', 'img', 'section', 'h1', 'h2', 'h3', 'ul', 'ol', 'li', 'div', 'article']
structural_tags = ['section', 'ul', 'ol'] #'head', 'body',
ignored_tags = ['tspan', 'textpath', ] 


def get_immediate_parents(tag):
    parents_list = []
    for parent in tag.find_parents():
        parents_list.append(parent)
        if parent.name == 'html': break
    return parents_list

# @param: soup -> soup object of the html page
# returns list the nodes of possible target labels
# fix: only returns html tag
def extract_target_labels(soup):
    # target_labels = []
    # for tag in soup.find_all():
    #     if tag.name == 'div' and len(tag.find_all(recursive=False)) > 1:
    #         target_labels.append(tag)
    #     elif tag.name in structural_tags:
    #         target_labels.append(tag)
    # return target_labels
    return [soup.html]


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
# returns array of contexts (-> strings of the form: 'tag1,path,tag2')
def extract_contexts(node, simple=False):
    if node == None: return []
    contexts = []
    for tag in node.find_all():
        if tag.name in tokens_vocab:
            if len(tag.find_all()) > 0: # ensure that node is not leaf
                for c in tag.find_all(recursive=not simple):
                    if c.name in ignored_tags: continue
                    p = extract_path(tag, c, node)
                    if p in contexts: continue                    
                    contexts.append(f"{tag.name},{p},{c.name}")
            elif tag.find_parent():
                p = extract_path(tag, tag.find_parent(), node)
                if p in contexts: continue
                contexts.append(f"{tag.name},{p},{tag.find_parent().name}")
                if tag.text and tag.text != '\n':
                    text = re.sub(r'\s+', '_', str(tag.text).replace('\n', '').replace('\t', '').replace(',', ''))
                    if text != '': contexts.append(f"{tag.name},contains," + text)
    return contexts

# prints rows as defined in the code2vec docs, -> final output for a row: 'target_label context1 context2 context3 ...'
# @param: soup -> soup object of the html page
def print_examples(soup):
    target_labels = extract_target_labels(soup)
    for target_label in target_labels:
        contexts = extract_contexts(target_label, simple=False)
        print(f'{target_label.name} {" ".join([con for con in contexts])}\n')    
        # print(f'{target_label.name}|{str(target_label.attrs).replace(" ", "")} {" ".join([con for con in contexts])}\n')    # add attributes of target label to output

# prints examples which are extracted from a single html file
# @param: file -> path to html file    
def extract(file):
    soup = BeautifulSoup(open(file), 'html.parser')
    print_examples(soup)

# prints the cosine similarity
# @param: contexts -> list of tupels of the form:(name, contexts) to compute the similarity for (each represents a different html site)
def print_cosine_similarity(contexts):
    names, representations = zip(*tuples)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(representations)
    cosine_similarities = cosine_similarity(tfidf_matrix)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            similarity = cosine_similarities[i][j]
            print(f"Similarity between {names[i]} and {names[j]}: {similarity}")

if __name__ == '__main__':
    parser = ArgumentParser()
    # parser.add_argument("-maxlen", "--max_path_length", dest="max_path_length", required=False, default=8)
    # parser.add_argument("-maxwidth", "--max_path_width", dest="max_path_width", required=False, default=2)
    # parser.add_argument("-threads", "--num_threads", dest="num_threads", required=False, default=64)
    # parser.add_argument("-file", "--file", dest="file", required=False)
    # parser.add_argument("-dir", "--dir", dest="dir", required=False)
    # args = parser.parse_args()
    # if args.file is not None:
    #     extract(args.file)
    # elif args.dir is not None:
    #     for file in os.listdir(args.dir):
    #         extract(os.path.join(args.dir, file))

    with open('code2vec/data/extractor_results.txt', 'r') as f:
        print(f.readline())
        original = f.readline()
        print(f.readline())
        deleted_clone = f.readline()
        print(f.readline())
        added_clone = f.readline()
        print(f.readline())
        altered_text_clone = f.readline()
        print(f.readline())
        reshuffle_clone = f.readline()
        print(f.readline())
        distinct_page = f.readline()
    tuples = [('original', original),
            ('distinct_page', distinct_page),
            ('deleted_clone', deleted_clone),
            ('added_clone', added_clone),
            ('altered_text_clone', altered_text_clone),
            ('reshuffle_clone', reshuffle_clone)
            ]
    print_cosine_similarity(tuples)
            

            

        

