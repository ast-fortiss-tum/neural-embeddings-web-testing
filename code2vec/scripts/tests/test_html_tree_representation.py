import unittest
import sys
sys.path.append('code2vec/scripts')
from bs4 import BeautifulSoup
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

if __name__ == '__main__':
    unittest.main()