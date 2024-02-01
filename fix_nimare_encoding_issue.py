"""
Fixes encoding issue of references.bib in nimare 0.2.0.
Requires `pip install unidecode`
"""
from nimare.utils import get_resource_path
import os
from unidecode import unidecode


with open(os.path.join(get_resource_path(), 'references.bib'), 'r', encoding='utf-8') as f:
    content = f.read()
ascii_content = unidecode(content)
with open(os.path.join(get_resource_path(), 'references.bib'), 'w', encoding='utf-8') as f:
    f.write(ascii_content)