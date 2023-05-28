"""
BASE_URLS: as url base foram obtidas da área do site "Área do Conhecimento".
"""
import os
from crawler.crawler import Crawler

DATA_PATH = "../data"
BASE_URL = "https://www.teses.usp.br/index.php?option=com_jumi&fileid=9&Itemid=159&lang=pt-br&id={}&prog={}&exp=0&pagina="  # noinspection PyCharm
BASE_URL_FORMAT_DATA = [
    ('59135', '59005'),
    ('17155', '17023'),
    ('17165', '17028'),
    ('98131', '98001'),
    ('5178', '5006'),
    ('5137', '5017'),
    ('17139', '17003'),
    ('98132', '98002'),
    ('337', '1'),
    ('94', '1')
]

crawler = Crawler(data_path=DATA_PATH)
for format_data in BASE_URL_FORMAT_DATA:
    crawler.add_base_url(BASE_URL.format(*format_data))
crawler.execute()
