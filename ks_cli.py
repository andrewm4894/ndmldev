import argparse
from urllib.parse import parse_qs, urlparse

from utils import get_chart_list

parser = argparse.ArgumentParser()
parser.add_argument(
    '--url', type=str, nargs='?', help='url', default='http://127.0.0.1:19999/'
)
args = parser.parse_args()

# parse args
url = args.url

baseline_window_multiplier = 2

url_params = parse_qs(url)
url_parse = urlparse(url)
host = url_parse.netloc
after = int(int(url_params.get('after')[0]) / 1000)
before = int(int(url_params.get('before')[0]) / 1000)
highlight_after = int(int(url_params.get('highlight_after')[0]) / 1000)
highlight_before = int(int(url_params.get('highlight_before')[0]) / 1000)
starts_with = None
window_size = highlight_before - highlight_after
baseline_before = highlight_after - 1
baseline_after = baseline_before - (window_size * baseline_window_multiplier)

charts = get_chart_list(starts_with=starts_with, host=host)
api_calls = [
    (f'http://{host}/api/v1/data?chart={chart}&after={baseline_after}&before={highlight_before}&format=json', chart)
    for chart in charts
]

print(api_calls)
