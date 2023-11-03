
# this script should test wheter the extractor for html pages is sufficiently robust to detect clones -> needed for representation mechanism of the code2vec model

from bs4 import BeautifulSoup, Tag, NavigableString
import random
from math import log
import os
import sys
import string
sys.path.append('code2vec/HTMLExtractor')
from utils import extract_contexts, calculate_cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

container_tags = ['div', 'article', 'section', 'ul', 'ol', 'div', 'div', 'div', 'div', 'div']
text_tags = ['p', 'a', 'button', 'form', 'img', 'h1', 'h2', 'h3']
number_of_childs = 3

STEP_LENGTH = 0.1
GENERATED_TEXT_MIN_LENGTH = 3
GENERATED_TEXT_MAX_LENGTH = 20

# util methods
# @param: soup -> BeautifulSoup object of the html file
# @return: tag_dict -> dict of the form {no_of_desc: [tag1, tag2, tag3, ...]} => key: number of descendants, value: list of tags with that number of descendants
#          sorted_keys -> list of keys of the dict, sorted in ascending order
def generate_numberofdesc_tag_dict(soup):
    tag_dict = {}
    for tag in soup.find_all():
        number_of_descendants = sum(1 for _ in tag.descendants if isinstance(_, str))
        if number_of_descendants in tag_dict:
            tag_dict[number_of_descendants].append(tag)
        else:
            tag_dict[number_of_descendants] = [tag]
    sorted_keys = sorted(tag_dict.keys())
    return tag_dict, sorted_keys

# function to geneate html 
# @param: depth -> depth of the generated html
# @return: elment -> parent element of the created html
def generate_html(depth, is_list=False, initial=False):
    if depth <= 0: print(f"Invalid depth: {depth}, no html added");return None
    if depth == 1:
        new_tag = Tag(name=random.choice(text_tags))
        new_tag.string = ''.join(random.choice(string.ascii_lowercase) for _ in range(GENERATED_TEXT_MIN_LENGTH, GENERATED_TEXT_MAX_LENGTH))
        return new_tag
    else:
        if initial:
            new_tag = random.choice(['div', 'article', 'section'])
            tag = Tag(name=new_tag)
            for _ in range(random.randint(number_of_childs - 1, number_of_childs)):
                child = generate_html(depth-1)
                tag.append(child)
            return tag 
        if is_list:
            new_tag = Tag(name=random.choice(['ul', 'ol']))
            for _ in range(random.randint(number_of_childs - 1, number_of_childs)):
                li = Tag(name='li')
                li.append(generate_html(depth-1))
                new_tag.append(li)
            return new_tag
        new_tag = random.choice(container_tags)
        if new_tag == 'ul' or new_tag == 'ol':
            return generate_html(depth, is_list=True)
        tag = Tag(name=new_tag)
        for _ in range(random.randint(number_of_childs - 1, number_of_childs)):
            child = generate_html(depth-1)
            tag.append(child)
        return tag 

# @param: tag -> tag object to shuffle its childs
def shuffle_childs(tag):
    children = list(tag.children)
    random.shuffle(children)
    tag.clear()
    for child in children:
        tag.append(child)


# operations to generate the different clones

# @param: no_desc_taglist_dict -> dict of the form {no_of_desc: [tag1, tag2, tag3, ...]} => key: number of descendants, value: list of tags with that number of descendants
# @param: keys -> list of keys of the dict, sorted in ascending order
# @param: intensity -> [0-1]: intensity of the mutation (how many elements should be deleted) => calculation via log(intensity * len(keys))
def generate_delete_structure_clone(no_desc_taglist_dict, keys, intensity):
    if intensity == 0: return
    elements_to_delete = []
    ix = round(intensity * len(keys))
    ix = ix if ix < len(keys) else len(keys) - 1
    for i, key in enumerate(keys[:ix+1]):
        for element in no_desc_taglist_dict[key]:
            if random.random() <= intensity and random.random()<= intensity:
                elements_to_delete.append(element)
    for element in elements_to_delete:
        element.decompose()
    # print(f"Remaining elements: {len(no_desc_taglist_dict[keys[-1]][0].find_all())}")

def generate_add_structure_clone(no_desc_taglist_dict, keys, intensity):
    if intensity == 0: return
    if(len(keys) < 2): print("Error while trying to generate Add-Structure Clone");return

    new_hmtl = generate_html(round(intensity*10), initial=True)
    if(not new_hmtl):return
    random.choice(no_desc_taglist_dict[random.choice((keys[1:]))]).append(new_hmtl)
    # print(f"Total elements: {len(no_desc_taglist_dict[keys[-1]][0].find_all())}")

def generate_alter_text_clone(soup, intensity):
    if intensity == 0: return
    all_text_tags = list(soup.find_all(text_tags))
    for element in all_text_tags:
        if not element.string: continue
        if random.random() < intensity:
            element.string=NavigableString(''.join(random.choice(string.ascii_letters) for _ in range(random.randint(GENERATED_TEXT_MIN_LENGTH, GENERATED_TEXT_MAX_LENGTH))))

def generate_reshuffle_clone(no_desc_taglist_dict, keys, intensity,):
    # print(f"Intensity: {intensity}")
    if intensity == 0: return
    ix = round(intensity * len(keys))
    ix = ix if ix < len(keys) else len(keys) - 1
    keys = keys[:ix]
    for key in keys:
        for element in no_desc_taglist_dict[key]:
            shuffle_childs(element)

# function to test the extractor using clone generation functions
# @param: filepath -> path to the html file
# @param: delete -> bool, default=False -> test with delete clone
# @param: add -> bool, default=False -> test with add clone
# @param: alter_text -> bool, default=False -> test with alter text clone
# @param: reshuffle -> bool, default=False -> test with reshuffle clone
def test_extractor_using_clone(filepath, delete=False, add=False, alter_text=False, reshuffle=False):
    conditions = [delete, add, alter_text, reshuffle]
    if sum(conditions) != 1: print("Error: Invalid conditions for test_extractor_using_clones, set only 1 to True"); return
    soup = BeautifulSoup(open(filepath), 'html.parser')
    original_contexts = extract_contexts(soup)
    tupels =[(f'Original({os.path.basename(filepath)})', " ".join(con for con in original_contexts))]
    intensity=0
    while intensity <= 1-STEP_LENGTH:
        soup_intens = BeautifulSoup(open(filepath), 'html.parser')
        if alter_text:
            clone_name = 'altered_text'
            generate_alter_text_clone(soup, intensity)
        else:
            clone_name = 'reshuffle' if reshuffle else 'deleted' if delete else 'added'
            clone_generation_function = generate_reshuffle_clone if reshuffle else generate_delete_structure_clone if delete else generate_add_structure_clone
            no_desc_taglist_dict, keys = generate_numberofdesc_tag_dict(soup_intens)
            clone_generation_function(no_desc_taglist_dict, keys, intensity)
        mutated_contexts = extract_contexts(soup_intens)
        tupels.append((f'{clone_name}_clone_{int(intensity*100)}%', " ".join([con for con in mutated_contexts])))
        print(calculate_cosine_similarity(tupels))
        tupels.pop(1)
        print(calculate_cosine_similarity_bare_html_representation(soup, soup_intens, f'Original({os.path.basename(filepath)})', f'{clone_name}_clone_{int(intensity*100)}%')) # add bare html for comparisions

        intensity += STEP_LENGTH

# function to test the extractor using clone generation functions => prints out the cosine similarity between the original and the clones for every type of clone
# @param: filepath -> path to the html file
# @param: comparison_filepaths -> list of paths to html files to compare the original with 
def test_extractor_with_generating_clones(filepath, comparison_filepaths=None, delete=True, add=True, alter_text=True, reshuffle=True):
    if comparison_filepaths:
        print(f'Calculating representation similarity with comparisions...\n')
        original_contexts = extract_contexts(BeautifulSoup(open(filepath), 'html.parser'))
        tupels =[(f'original({os.path.basename(filepath)})', " ".join(con for con in original_contexts))]
        for i, comparison_filepath in enumerate(comparison_filepaths):
            comparison_contexts = extract_contexts(BeautifulSoup(open(comparison_filepath), 'html.parser'))
            tupels.append((f"Comparision {i} ({os.path.basename(comparison_filepath)})", " ".join(con for con in comparison_contexts)))
            print(calculate_cosine_similarity(tupels))
            tupels.pop(1)
        print('==========\n')

    print("Testing extractor with generating clones...\n")
    if(delete): 
        print('[CLONE-TYPE: DELETED]')
        test_extractor_using_clone(filepath, delete=True)
    if add:
        print('\n[CLONE-TYPE: ADDED]')
        test_extractor_using_clone(filepath, add=True)
    if alter_text:
        print('\n[CLONE-TYPE: ALTERED TEXT]')
        test_extractor_using_clone(filepath, alter_text=True)
    if reshuffle:
        print('\n[CLONE-TYPE: RESHUFFLE]')
        test_extractor_using_clone(filepath, reshuffle=True)
    


# function to calculate the cosine similarity between 2 representations of html pages using tdidf
def calculate_cosine_similarity_bare_html_representation(soup1, soup2, name1, name2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([soup1.prettify(), soup2.prettify()])
    cosine_similarities = cosine_similarity(tfidf_matrix)
    return(f"Similarity between {name1} and {name2}: {round(cosine_similarities[0, 1], 4)} =>(bare html)")


if __name__ == '__main__':    

    # print('MDN Webdocs')
    # test_extractor_with_generating_clones('code2vec/resources/MDN_webdocs.html', comparison_filepaths=['code2vec/resources/MDN_Blog.html', 'code2vec/resources/Wolfram_Alpha.html'])

    print('Github')
    test_extractor_with_generating_clones('code2vec/resources/GitHub-Homepage.html', comparison_filepaths=['code2vec/resources/GitHub-Pricing.html', 'code2vec/resources/Wolfram_Alpha.html'])