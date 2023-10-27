
# this script should test wheter the extractor for html pages is sufficiently robust to detect clones -> needed for representation mechanism of the code2vec model

from bs4 import BeautifulSoup
import random
from math import log
import os
from code2vec.HTMLExtractor.utils import extract_contexts, print_cosine_similarity


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

# operations to generate the different clones

# @param: no_desc_taglist_dict -> dict of the form {no_of_desc: [tag1, tag2, tag3, ...]} => key: number of descendants, value: list of tags with that number of descendants
# @param: keys -> list of keys of the dict, sorted in ascending order
# @param: intensity -> [0-1]: intensity of the mutation (how many elements should be deleted) => calculation via log(intensity * len(keys))
def generate_delete_structure_clone(no_desc_taglist_dict, keys, intensity, num_elems=0):
    print(f"Intensity: {intensity}")
    if intensity == 0: return
    elements_to_delete = []
    ix = round(log(intensity * len(keys)))
    ix = ix if ix < len(keys) else len(keys) - 1
    for i, key in enumerate(keys[:ix+1]):
        for element in no_desc_taglist_dict[key]:
            if random.random() <= intensity:
                elements_to_delete.append(element)
    count = 0
    for element in elements_to_delete:
        element.decompose()
        count += 1
    print(f"Deleted {count} elements.")

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
    tupels =[('original', original_contexts),]
    for i in range(0,11):
        soup_intens = BeautifulSoup(open(filepath), 'html.parser')
        intensity = i / 10
        no_desc_taglist_dict, keys = generate_numberofdesc_tag_dict(soup_intens)
        generate_delete_structure_clone(no_desc_taglist_dict, keys, intensity)
        mutated_contexts = extract_contexts(soup_intens)
        tupels.append((f"deleted_clone_{intensity}", mutated_contexts))

    print_cosine_similarity(tupels)
        


if __name__ == '__main__':
    test_extractor_deleted_clones('code2vec/resources/MDN_webdocs.html')









    # html_content = """
    # <html>
    # <body>
    #     <div>
    #         <p>Paragraph 1</p>
    #         <p>Paragraph 2</p>
    #     </div>
    #     <span>Span</span>
    # </body>
    # </html>
    # """
    # print(f"Original doc: {len(BeautifulSoup(open('C:/Users/lucaa/Uni/swq23-neural-embeddings-web-testing/code2vec/resources/MDN_webdocs.html'), 'html.parser').find_all())} number of tags")
    # for i in range(0, 11):
    #     # soup = BeautifulSoup(html_content, 'html.parser')
    #     soup = BeautifulSoup(open('C:/Users/lucaa/Uni/swq23-neural-embeddings-web-testing/code2vec/resources/MDN_webdocs.html'), 'html.parser')
    #     intensity = i / 10
    #     no_desc_taglist_dict, keys = generate_numberofdesc_tag_dict(soup)
    #     generate_delete_structure_clone(no_desc_taglist_dict, keys, intensity)
    #     print(f"number of tags after mutation: {len(soup.find_all())}")
    #     # print("HTML after mutation: ")
    #     # print(soup.prettify())