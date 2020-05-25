import requests


def get_chart_data_urls():
    url = "https://127.0.0.1/api/v1/charts"
    r = requests.get(url)
    charts = r.json().get('charts')
    chart_data_urls = [charts[chart].get('data_url') for chart in charts]
    return chart_data_urls

