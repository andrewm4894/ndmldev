import requests


def get_chart_data_urls():
    url = "http://127.0.0.1:19999/api/v1/charts"
    r = requests.get(url)
    charts = r.json().get('charts')
    chart_data_urls = [charts[chart].get('data_url') for chart in charts]
    return chart_data_urls

