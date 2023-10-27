
# this script should test wheter the extractor for html pages is sufficiently robust to detect clones -> needed for representation mechanism of the code2vec model

from bs4 import BeautifulSoup, Tag
import random
from math import log
import os
import sys
import string
sys.path.append('code2vec/HTMLExtractor')
from utils import extract_contexts, print_cosine_similarity

container_tags = ['div', 'article', 'section', 'ul', 'ol']
text_tags = ['p', 'a', 'button', 'form', 'img', 'h1', 'h2', 'h3']

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
# @param: number_of_elements -> number of elements to generate
# @param: depth -> depth of the generated html
# @return: elment _> parent element of the created html
def generate_html(number_of_elements, depth, is_list=False):
    if number_of_elements <= 0 or depth < 0: print("Invalid number of elements/depth\n");return None
    if number_of_elements == 1:
        new_tag = Tag(name=random.choice(text_tags))
        new_tag.string = ''.join(random.choice(string.ascii_lowercase) for _ in range(10))
        return new_tag
    else:
        new_tag = random.choice(container_tags)
        

        return new_tag
    



# operations to generate the different clones

# @param: no_desc_taglist_dict -> dict of the form {no_of_desc: [tag1, tag2, tag3, ...]} => key: number of descendants, value: list of tags with that number of descendants
# @param: keys -> list of keys of the dict, sorted in ascending order
# @param: intensity -> [0-1]: intensity of the mutation (how many elements should be deleted) => calculation via log(intensity * len(keys))
def generate_delete_structure_clone(no_desc_taglist_dict, keys, intensity, num_elems=0):
    print(f"Intensity: {intensity}")
    if intensity == 0: return
    elements_to_delete = []
    ix = round(intensity * len(keys))
    ix = ix if ix < len(keys) else len(keys) - 1
    for i, key in enumerate(keys[:ix+1]):
        for element in no_desc_taglist_dict[key]:
            if random.random() <= intensity:
                elements_to_delete.append(element)
    for element in elements_to_delete:
        element.decompose()
    print(f"Remaining elements: {len(no_desc_taglist_dict[keys[-1]][0].find_all())}")

def generate_add_structure_clone(no_desc_taglist_dict, keys, intensity):
    if intensity == 0: return
    
    pass

def generate_alter_text_clone():
    pass

def generate_reshuffle_clone():
    pass

# todo: implement, maybe in utlis.py?
# steps: => (the same for all clones) 
#   1. create a clone with an intensity
#   2. extract the paths, extract(soup=soup)
#   3. use print_cosinine_similarity to compute() the similarity between the original and the mutated html file
#   4. look at similarity, at some point it should be very far away from the original => stop eventually
#   5. increase the intensity and repeat step 1

def test_extractor_deleted_clones(filepath):
    soup = BeautifulSoup(open(filepath), 'html.parser')
    original_contexts = extract_contexts(soup)
    tupels =[('original', " ".join(con for con in original_contexts)),]

    for i in range(0,21, 1):
        soup_intens = BeautifulSoup(open(filepath), 'html.parser')
        intensity = (i * 0.5) / 10
        no_desc_taglist_dict, keys = generate_numberofdesc_tag_dict(soup_intens)
        generate_delete_structure_clone(no_desc_taglist_dict, keys, intensity)
        mutated_contexts = extract_contexts(soup_intens)
        tupels.append((f'deleted_clone_{int(intensity*100)}%', " ".join([con for con in mutated_contexts])))
    print_cosine_similarity(tupels, use_only_first_forcomp=True, add_comparisons=True)

#TODO: implement the other clones
def test_extractor_added_clones(filepath):
    pass

def test_extractor_altered_text_clones(filepath):
    pass

def test_extractor_reshuffle_clones(filepath):
    pass
        

if __name__ == '__main__':
    # test_extractor_deleted_clones('code2vec/resources/MDN_webdocs.html')

    html_content = """
<html>
<head>
    <title>Sample Page</title>
</head>
<body>
    <h1>Hello, World!</h1>
</body>
</html>
"""

# Parse the HTML content using Beautiful Soup
soup = BeautifulSoup(html_content, 'html.parser')
# Create a new <div> element
# Find the <body> tag and append the new <div> element to it
body_tag = soup.body
body_tag.append(generate_html(1, 1))
# Print the modified HTML content
print(soup.prettify())
