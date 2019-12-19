import requests
from abc import ABC
import json
import csv

from csv_handler import CSVHandler

BASE_API_URL = 'https://api-fxpractice.oanda.com'
BASE_STREAM_URL = 'https://stream-fxpractice.oanda.com'
MAX_RETRIEVE = 30


class Oanda(ABC):
    """Oanda API handler class
    Usage::


    """

    def __init__(self, access_token, account_id):
        self.access_token = access_token
        self.account_id = account_id


class CandlePricingAPI(Oanda):
    """
    API to get the current pricing

    """

    def __init__(self, access_token, account_id, csv_path):
        super().__init__(access_token, account_id)
        self.csv_handler = CSVHandler(csv_path)

    def get_candle_price(self):
        url = BASE_API_URL + '/v3/accounts/' + \
            self.account_id + '/candles/latest'
        res = requests.get(url,
                           headers={'Authorization': 'Bearer ' +
                                    self.access_token},
                           params={'candleSpecifications': 'EUR_USD:M1:B'})

        if(res.status_code != 200):
            # need to handle error
            pass

        data = json.loads(res.text)
        row_data = data['latestCandles'][0]['candles'][-1]['bid'].values()
        self.csv_handler.save_to_csv(row_data)


class StreamListener:
    def __init__(self, lstm_model, csv_path):
        self.counter = 0
        self.csv_handler = CSVHandler(csv_path)
        self.lstm_model = lstm_model

    def on_data(self, data):
        if(self.counter > MAX_RETRIEVE):
            return False
        else:
            self.handle_data(data)
            return True

    def handle_data(self, data):
        try:
            price = data['bids'][0]['price']
            time = data['time']
            row_data = [time, price]
            self.csv_handler.save_to_csv(row_data)
        except Exception as e:
            raise e


class PricingStreamAPI(Oanda):
    """
    API to get streaming pricing
    """

    def __init__(self, access_token, account_id, listener):
        super().__init__(access_token, account_id)
        self.listener = listener

    def get_stream_price(self):
        url = BASE_STREAM_URL + '/v3/accounts/' + \
            self.account_id + '/pricing/stream'
        try:
            it = requests.get(url,
                              headers={'Authorization': 'Bearer ' +
                                       self.access_token},
                              params={'instruments': 'USD_JPY',
                                      'snapshot': True},
                              stream=True)

            if(it.status_code != 200):
                """
                handle of status code not 200
                """
                pass

            for line in it.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    data = json.loads(decoded_line)
                    if(data['type'] == 'PRICE'):
                        self.listener.on_data(data)

        except Exception as e:
            raise e


candle_api = CandlePricingAPI(
    'f3ebd9b6f7db45248ed34c73f6047a37-c464a1f9b3f980b240b6b0008c928b3c', '101-011-12884650-001', './eur_usd.csv')
candle_api.get_candle_price()
