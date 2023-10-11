import unittest
from bs4 import BeautifulSoup
import sys
sys.path.append('code2vec/scripts')
from html_tree_representation import extract_path

class test_html_tree_representation(unittest.TestCase):
    def test_extract_path_basic(self):
        basic_html_content = """
        <html>
        <head>
            <title>Sample Page</title>
        </head>
        <body>
            <div>
                <p>Paragraph inside div</p>
            </div>
            <button>
                <p>Paragraph inside button</p>
            </button>
        </body>
        </html>
        """

        soup = BeautifulSoup(basic_html_content, "html.parser")

        simple_path_direct_parent = extract_path(soup.body.div.p, soup.div, soup)
        self.assertEqual(simple_path_direct_parent, "^")
        simple_path_direct_child = extract_path(soup.head, soup.head.title, soup)
        self.assertEqual(simple_path_direct_child, "_")
        simple_path_direct_sibling = extract_path(soup.body.div, soup.body.button, soup)
        self.assertEqual(simple_path_direct_sibling, "^body_")
        simple_path_common_ancestor_html = extract_path(soup.body.div.p, soup.head.title, soup)
        self.assertEqual(simple_path_common_ancestor_html, "^div^body^html_head_")
        simple_path_common_ancestor_body = extract_path(soup.body.button.p ,soup.body.div.p, soup)
        self.assertEqual(simple_path_common_ancestor_body, "^button^body_div_")

    def test_extract_path_long_paths(self):
        html_path = "code2vec/testHTML/MDN_webdocs.html"
        with open(html_path, 'r', encoding='utf-8') as html_file: html_content = html_file.read()
        soup = BeautifulSoup(html_content, 'html.parser')
        long_path_common_ancestor_html = extract_path(soup.head.title, soup.body.div.div.div.section.p, soup)
        self.assertEqual(long_path_common_ancestor_html, "^head^html_body_div_div_div_section_")
        long_path_common_ancestor_body = extract_path(soup.body.div.footer.div.div.p, soup.body.find_all(name='div', recursive=False)[-1], soup)
        self.assertEqual(long_path_common_ancestor_body, "^div^div^footer^div^body_")
        page_footer_moz_div = soup.find_all('div')[-3]
        long_path_only_first_parent = extract_path(page_footer_moz_div, page_footer_moz_div.a.title, soup)
        self.assertEqual(long_path_only_first_parent, "_a_svg_")
        long_path_only_second_parent = extract_path(page_footer_moz_div.a.title, page_footer_moz_div, soup)
        self.assertEqual(long_path_only_second_parent, "^svg^a^")

    def test_extract_path_edge_cases(self):
        todo = True

        
if __name__ == '__main__':
    unittest.main()