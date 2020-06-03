import argparse
from urllib.parse import parse_qs

parser = argparse.ArgumentParser()
parser.add_argument(
    '--url', type=str, nargs='?', help='url', default='http://127.0.0.1:19999/'
)
args = parser.parse_args()

# parse args
url = args.url

url_params = parse_qs(url)
after = url_params.get('after')
before = url_params.get('before')
highlight_after = url_params.get('highlight_after')
highlight_before = url_params.get('highlight_before')

print(url)
