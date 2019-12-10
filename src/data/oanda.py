import requests
from abc import ABC, abstractmethod


BASE_URL = ''
BASE_STREAM_URL = ''


class Oanda:
    """Oanda API handler class
    Usage::


    """

    def __init__(self, access_token, account_id):
        self.access_token = access_token
        self.account_id = account_id

    def get_stream_price(self):
        it = requests.get(BASE_STREAM_URL + '/v3/accounts/' +
                          self.account_id + '/pricing',
                          params={'instruments': 'USD_JPY'},
                          headers='Authorization: Bearer ' + self.access_token,
                          stream=True)
        for line in it.iter_lines():
            print(line)


oanda_api = Oanda(
    'f3ebd9b6f7db45248ed34c73f6047a37-c464a1f9b3f980b240b6b0008c928b3c', '101-011-12884650-001')

oanda_api.get_stream_price()
