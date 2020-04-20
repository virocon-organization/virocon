import requests
from bs4 import BeautifulSoup

URL = 'https://apps.ecmwf.int/datasets/data/interim-full-daily/levtype=sfc/?date_year_month=201104&time=00:00:00,06:00:00,12:00:00,18:00:00&step=0&param=232.140,229.140'

page = requests.get(URL)

soup = BeautifulSoup(page.content, 'html.parser')
print(soup.prettify())





