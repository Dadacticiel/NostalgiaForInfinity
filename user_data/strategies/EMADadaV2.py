# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
from freqtrade.strategy import IntParameter, DecimalParameter
from functools import reduce
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import pandas as pd
from freqtrade.resolvers import ExchangeResolver
from freqtrade.persistence import Trade

class EMADadaV2(IStrategy):
    INTERFACE_VERSION = 3

    # Define the parameters of the strategy
    timeframe = '4h'
    startup_candle_count = 30
    max_open_trades = 2
    stoploss = -0.6
    process_only_new_candles = True

    # Define the lengths of the EMAs
    buy_ema_1_len = IntParameter(1, 19, default=15, space='buy')
    buy_ema_2_len = IntParameter(20, 60, default=20, space='buy')
    buy_candle_length = IntParameter(5, 30, default=14, space='buy')
    sell_candle_threshold = IntParameter(50, 99, default=80, space='sell')
    buy_candle_smooth = IntParameter(5, 40, default=20, space='buy')

    # Buy hyperspace params:
    buy_params = {
        "rsi_entry_long": 27,
        "rsi_entry_short": 59,
        "window": 24,
    }

    # Sell hyperspace params:
    sell_params = {
        "rsi_exit_long": 18,
        "rsi_exit_short": 75,
    }

    # rsi_entry_long  = IntParameter(0, 100, default=buy_params.get('rsi_entry_long'),  space='buy',  optimize=True)
    # rsi_exit_long   = IntParameter(0, 100, default=buy_params.get('rsi_exit_long'),   space='sell', optimize=True)
    # window          = IntParameter(5, 100, default=buy_params.get('window'),          space='buy',  optimize=False)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        # dataframe['rsi_ema'] = dataframe['rsi'].ewm(span=self.window.value).mean()

        # # Ensure 'rsi_ema' does not have NaN values before calculating the gradient
        # dataframe['rsi_ema'] = dataframe['rsi_ema'].bfill()  # Backward fill

        # # Check if there are enough values to calculate the gradient
        # if len(dataframe['rsi_ema']) > 1:
        #     dataframe['rsi_gra'] = np.gradient(dataframe['rsi_ema'])
        # else:
        #     dataframe['rsi_gra'] = 0  # Or handle differently

        # Calculate the EMAs
        dataframe['buy_ema_1'] = ta.EMA(dataframe, timeperiod=self.buy_ema_1_len.value)
        dataframe['buy_ema_2'] = ta.EMA(dataframe, timeperiod=self.buy_ema_2_len.value)
        dataframe['buy_ema_low'] = ta.EMA(dataframe['low'], timeperiod=self.buy_ema_2_len.value)

        # Calculate additional conditions
        dataframe['going_up'] = (dataframe['buy_ema_1'].shift(5) < dataframe['buy_ema_1']) & \
                                (dataframe['buy_ema_2'].shift(5) < dataframe['buy_ema_2']) & \
                                (dataframe['buy_ema_1'].shift(1) < dataframe['buy_ema_1']) & \
                                (dataframe['buy_ema_2'].shift(1) < dataframe['buy_ema_2']) & \
                                (((dataframe['buy_ema_1'] / dataframe['buy_ema_1'].shift(5)) - 1) > 0.0003)
        dataframe['close_superior_to_previous_high'] = dataframe['high'].shift(6) > dataframe['close']

        # Paramètres de l'ADX
        len_dmi = 17
        lensig = 14

        # Calcul de DMI et ADX
        dataframe['diplus'] = ta.PLUS_DI(dataframe, timeperiod=len_dmi)
        dataframe['diminus'] = ta.MINUS_DI(dataframe, timeperiod=len_dmi)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=lensig)

        # exchange = ExchangeResolver().load_exchange(config=self.config)
        # market = exchange.markets.get(metadata['pair'])
        # if market:
        #     tick_size = market['precision']['price']
        #     dataframe['min_tick'] = tick_size
        # else:
        dataframe['min_tick'] = 0.001  # Handle cases where market information is not found

        new_df = pd.DataFrame({
            'close': dataframe['close'],
            'high': dataframe['close'],
            'low': dataframe['close']
        })
        stoch_values = self.stochastic_oscillator(new_df, self.buy_candle_length.value)
        dataframe['k'] = stoch_values

        # Calcul initial de alpha basé sur le smooth
        alpha = 2 / (self.buy_candle_smooth.value + 1)

        # Initialisation de smooth_k
        dataframe['smooth_k'] = np.nan  # Définir NaN pour tous d'abord

        # Première valeur valide de smooth_k basée sur la première valeur de 'k'
        first_valid_index = dataframe['k'].first_valid_index()

        # Vérifier si first_valid_index n'est pas None
        if first_valid_index is not None:
            dataframe.at[first_valid_index, 'smooth_k'] = dataframe.at[first_valid_index, 'k']

            # Itérer à travers les lignes pour simuler la logique persistante var de Pine Script
            for idx in range(first_valid_index + 1, len(dataframe)):
                previous_smooth_k = dataframe.at[idx - 1, 'smooth_k']
                current_k = dataframe.at[idx, 'k']
                if current_k > 50:
                    dataframe.at[idx, 'smooth_k'] = previous_smooth_k + (100 - previous_smooth_k) * alpha
                elif current_k < 50:
                    dataframe.at[idx, 'smooth_k'] = previous_smooth_k + (0 - previous_smooth_k) * alpha
                else:
                    dataframe.at[idx, 'smooth_k'] = previous_smooth_k  # Utilisation de la valeur précédente si k n'est ni > 50 ni < 50
        else:
            # S'il n'y a pas d'indices valides, vous pouvez choisir de remplir avec des valeurs par défaut ou d'ignorer
            dataframe['smooth_k'] = np.nan  # Remplit toute la colonne avec NaN

        # Conditions bullish et bearish
        dataframe['is_bullish'] = (dataframe['k'] >= self.sell_candle_threshold.value) & (dataframe['smooth_k'] >= self.sell_candle_threshold.value)
        dataframe['is_bearish'] = (dataframe['k'] <= (100 - self.sell_candle_threshold.value)) & (dataframe['smooth_k'] <= (100 - self.sell_candle_threshold.value))

        # Indicateurs pour la bougie actuelle
        dataframe['red_candle'] = dataframe['close'] < dataframe['open']
        dataframe['green_candle'] = dataframe['close'] > dataframe['open']

        # Mesures de la bougie
        dataframe['c_top'] = dataframe[['open', 'close']].max(axis=1)
        dataframe['c_bot'] = dataframe[['open', 'close']].min(axis=1)
        dataframe['hl_width'] = dataframe['high'] - dataframe['low']
        dataframe['bod_width'] = dataframe['c_top'] - dataframe['c_bot']
        dataframe['hw_per'] = ((dataframe['high'] - dataframe['c_top']) / dataframe['hl_width']) * 100
        dataframe['lw_per'] = ((dataframe['c_bot'] - dataframe['low']) / dataframe['hl_width']) * 100
        dataframe['b_per'] = (dataframe['bod_width'] / dataframe['hl_width']) * 100
        dataframe['doji'] = np.isclose(dataframe['close'], dataframe['open'], atol=dataframe['min_tick'])

        dataframe['doji'] = dataframe['doji'].fillna(False).astype(bool)

        # Patterns de chandeliers Bullish
        dataframe['hammer'] = dataframe['is_bearish'] & (dataframe['lw_per'] > (dataframe['b_per'] * 2)) & (dataframe['b_per'] < 50) & (dataframe['hw_per'] < 2) & (~dataframe['doji'])
        dataframe['inv_hammer'] = dataframe['is_bearish'] & (dataframe['hw_per'] > (dataframe['b_per'] * 2)) & (dataframe['b_per'] < 50) & (dataframe['lw_per'] < 2) & (~dataframe['doji'])
        dataframe['rising_3'] = dataframe['is_bearish'].shift(4) & (
            dataframe['green_candle'].shift(4) & (dataframe['b_per'].shift(4) > 50) &
            dataframe['red_candle'].shift(3) & (dataframe['c_top'].shift(3) <= dataframe['high'].shift(4)) & (dataframe['c_bot'].shift(3) >= dataframe['low'].shift(4)) &
            dataframe['red_candle'].shift(2) & (dataframe['c_top'].shift(2) <= dataframe['high'].shift(4)) & (dataframe['c_bot'].shift(2) >= dataframe['low'].shift(4)) &
            dataframe['red_candle'].shift(1) & (dataframe['c_top'].shift(1) <= dataframe['high'].shift(4)) & (dataframe['c_bot'].shift(1) >= dataframe['low'].shift(4)) &
            dataframe['green_candle'] & (dataframe['close'] > dataframe['high'].shift(4)) & (dataframe['b_per'] > 50)
        )

        dataframe['bull_engulfing'] = dataframe['is_bearish'].shift(1) & (
            dataframe['red_candle'].shift(1) & dataframe['green_candle'] & 
            (dataframe['bod_width'] > (dataframe['bod_width'].shift(1) / 2)) &
            (dataframe['open'] < dataframe['close'].shift(1)) & (dataframe['c_top'] > dataframe['c_top'].shift(1)) &
            (dataframe['rising_3'] == False) & (dataframe['doji'].shift(1) == False)
        )

        dataframe['three_white_soldiers'] = dataframe['is_bearish'].shift(3) & \
            ((dataframe['green_candle'].shift(2) & (dataframe['b_per'].shift(2) > 70)) &
            (dataframe['green_candle'].shift(1) & (dataframe['b_per'].shift(1) > 70) & (dataframe['c_bot'].shift(1) >= dataframe['c_bot'].shift(2)) & (dataframe['c_bot'].shift(1) <= dataframe['c_top'].shift(2)) & (dataframe['close'].shift(1) > dataframe['high'].shift(2))) &
            (dataframe['green_candle'] & (dataframe['b_per'] > 70) & (dataframe['c_bot'] >= dataframe['c_bot'].shift(1)) & (dataframe['c_bot'] <= dataframe['c_top'].shift(1)) & (dataframe['close'] > dataframe['high'].shift(1))))

        dataframe['morning_star'] = dataframe['is_bearish'] & \
            ((dataframe['red_candle'].shift(2) & (dataframe['b_per'].shift(2) > 80)) &
            (dataframe['red_candle'].shift(1) & (dataframe['bod_width'].shift(1) < (dataframe['bod_width'].shift(2) / 2)) & (dataframe['open'].shift(1) < dataframe['close'].shift(2))) &
            (dataframe['green_candle'] & (dataframe['close'] > (dataframe['high'].shift(2) + dataframe['low'].shift(2)) / 2)))

        dataframe['bull_harami'] = dataframe['is_bearish'] & \
            (dataframe['green_candle'] & (dataframe['high'] <= dataframe['c_top'].shift(1)) & (dataframe['low'] >= dataframe['c_bot'].shift(1)) & dataframe['red_candle'].shift(1))

        # dataframe['tweezer_bottom'] = dataframe['is_bearish'].shift(1) & \
        #     ((abs(dataframe['low'] - dataframe['low'].shift(1)) < dataframe['min_tick']) & dataframe['green_candle'] & dataframe['red_candle'].shift(1))

        # Bearish patterns
        dataframe['shooting_star'] = dataframe['is_bullish'] & (
            (dataframe['hw_per'] > 65) & (dataframe['b_per'] < 50) & (dataframe['lw_per'] < 10) & (~dataframe['doji']) &
            dataframe['green_candle'].shift(1) & dataframe['green_candle'].shift(2) & dataframe['green_candle'].shift(3)
        )

        dataframe['hanging_man'] = (
            (dataframe['lw_per'] > (dataframe['b_per'] * 2)) & (dataframe['b_per'] < 50) & (dataframe['hw_per'] < 2) & (~dataframe['doji'])
        )

        dataframe['falling_three'] = (
            dataframe['red_candle'].shift(4) & (dataframe['b_per'].shift(4) > 50) &
            dataframe['green_candle'].shift(3) & (dataframe['c_top'].shift(3) <= dataframe['high'].shift(4)) & (dataframe['c_bot'].shift(3) >= dataframe['low'].shift(4)) &
            dataframe['green_candle'].shift(2) & (dataframe['c_top'].shift(2) <= dataframe['high'].shift(4)) & (dataframe['c_bot'].shift(2) >= dataframe['low'].shift(4)) &
            dataframe['green_candle'].shift(1) & (dataframe['c_top'].shift(1) <= dataframe['high'].shift(4)) & (dataframe['c_bot'].shift(1) >= dataframe['low'].shift(4)) &
            dataframe['red_candle'] & (dataframe['close'] < dataframe['low'].shift(4)) & (dataframe['b_per'] > 50)
        )

        dataframe['bear_engulfing'] = dataframe['is_bullish'].shift(1) & (
            dataframe['green_candle'].shift(1) & dataframe['red_candle'] & 
            (dataframe['bod_width'] > (dataframe['bod_width'].shift(1) * 2)) & (dataframe['open'] > dataframe['close'].shift(1)) & 
            (dataframe['c_bot'] < dataframe['c_bot'].shift(1)) & (~dataframe['falling_three']) & (dataframe['doji'].shift(1) == False)
        )

        dataframe['three_black_crows'] = (
            dataframe['red_candle'].shift(2) & (dataframe['b_per'].shift(2) > 70) &
            dataframe['red_candle'].shift(1) & (dataframe['b_per'].shift(1) > 70) & (dataframe['c_top'].shift(1) <= dataframe['c_top'].shift(2)) & (dataframe['close'].shift(1) < dataframe['low'].shift(2)) &
            dataframe['red_candle'] & (dataframe['b_per'] > 70) & (dataframe['c_top'] <= dataframe['c_top'].shift(1)) & (dataframe['close'] < dataframe['low'].shift(1))
        )

        dataframe['evening_star'] = dataframe['is_bullish'] & (
            dataframe['green_candle'].shift(2) & (dataframe['b_per'].shift(2) > 80) &
            dataframe['green_candle'].shift(1) & (dataframe['bod_width'].shift(1) < (dataframe['bod_width'].shift(2) / 2)) & (dataframe['open'].shift(1) > dataframe['close'].shift(2)) &
            dataframe['red_candle'] & (dataframe['close'] < ((dataframe['high'].shift(2) + dataframe['low'].shift(2)) / 2))
        )

        dataframe['bear_harami'] = dataframe['is_bullish'] & (
            dataframe['red_candle'] & (dataframe['high'] <= dataframe['c_top'].shift(1)) & (dataframe['low'] >= dataframe['c_bot'].shift(1)) & dataframe['green_candle'].shift(1)
        )

        dataframe['tweezer_top'] = dataframe['is_bullish'].shift(1) & (
            np.isclose(dataframe['high'], dataframe['high'].shift(1), atol=0.01) & dataframe['red_candle'] & dataframe['green_candle'].shift(1)
        )

        # Dada Patterns
        # dataframe['wait_for_bounce'] = ((dataframe['close'] > dataframe['open']) & \
        #                                ((dataframe['close'] - dataframe['open']) > ((dataframe['high'].shift(1) - dataframe['low'].shift(1)) * 2.5)))
        dataframe['wait_for_bounce'] = False & ((dataframe['close'] > dataframe['open']) & \
                                       ((dataframe['close'] - dataframe['open']) > ((dataframe['high'].shift(1) - dataframe['low'].shift(1)) * 2.5))) | \
                                       (dataframe['green_candle'].shift(1) & ((dataframe['bod_width'] / dataframe['close']) > 0.02) & dataframe['red_candle'])
    
        dataframe['dada_back_on_track'] = False & (dataframe['is_bearish'].shift(1)) & \
                                  (dataframe['is_bearish'].shift(2)) & \
                                  (dataframe['is_bearish'].shift(3)) & \
                                  (dataframe['red_candle'].shift(1)) & \
                                  (dataframe['red_candle'].shift(2)) & \
                                  (dataframe['red_candle'].shift(3)) & \
                                  (dataframe['green_candle']) & \
                                  (dataframe['bod_width'] > (dataframe['bod_width'].shift(1) + dataframe['bod_width'].shift(2))) & \
                                  ((dataframe['low'].shift(1) < dataframe['low']) | (dataframe['low'].shift(2) < dataframe['low'])) & \
                                  (dataframe['bod_width'].shift(3) > (dataframe['bod_width'].shift(2) + dataframe['bod_width'].shift(1)))

        dataframe['dada_pattern'] = (
            dataframe['is_bearish'].shift(1) &
            (dataframe['b_per'].shift(2) < 20) &
            (dataframe['bod_width'] > dataframe['bod_width'].shift(1)) &
            (dataframe['b_per'].shift(1) > 69) &
            dataframe['green_candle'] &
            dataframe['red_candle'].shift(1) &
            dataframe['red_candle'].shift(2) &
            dataframe['red_candle'].shift(3) &
            (dataframe['bod_width'] / dataframe['close'] > 0.05)
        )

        dataframe['candle_bullish_pattern'] = (metadata['pair'] != 'PEPE/USDT') & (metadata['pair'] != 'SOL/USDT') & (metadata['pair'] != 'SOL/USDT') & (dataframe['is_bullish'] | dataframe['bull_engulfing'] | dataframe['hammer'] | dataframe['inv_hammer'] | dataframe['rising_3'] | dataframe['three_white_soldiers'] | dataframe['morning_star'] | dataframe['bull_harami'])

        dataframe['all_down'] = (
            (dataframe['buy_ema_1'].shift(1) > dataframe['buy_ema_1']) &
            (dataframe['buy_ema_2'].shift(1) > dataframe['buy_ema_2']) &
            (dataframe['buy_ema_1'].shift(2) > dataframe['buy_ema_1'].shift(1)) &
            (dataframe['buy_ema_2'].shift(2) > dataframe['buy_ema_2'].shift(1)) &
            (dataframe['buy_ema_1'].shift(3) > dataframe['buy_ema_1'].shift(2)) &
            (dataframe['buy_ema_2'].shift(3) > dataframe['buy_ema_2'].shift(2)) &
            (dataframe['buy_ema_1'].shift(4) > dataframe['buy_ema_1'].shift(3)) &
            (dataframe['buy_ema_2'].shift(4) > dataframe['buy_ema_2'].shift(3)) &
            (dataframe['buy_ema_1'].shift(5) > dataframe['buy_ema_1'].shift(4)) &
            (dataframe['buy_ema_2'].shift(5) > dataframe['buy_ema_2'].shift(4)) &
            (dataframe['buy_ema_1'].shift(10) > dataframe['buy_ema_1']) &
            (dataframe['buy_ema_2'].shift(10) > dataframe['buy_ema_2']) &
            (dataframe['buy_ema_1'].shift(6) > dataframe['buy_ema_1'].shift(5)) &
            (dataframe['buy_ema_2'].shift(6) > dataframe['buy_ema_2'].shift(5)) &
            (dataframe['buy_ema_1'].shift(7) > dataframe['buy_ema_1'].shift(6)) &
            (dataframe['buy_ema_2'].shift(7) > dataframe['buy_ema_2'].shift(6)) &
            (
                dataframe['going_up'].shift(1) |
                dataframe['going_up'].shift(2) |
                dataframe['going_up'].shift(3) |
                dataframe['going_up'].shift(4) |
                dataframe['going_up'].shift(5) |
                (
                    (metadata['pair'] == 'BTC/USDT') & 
                    dataframe['candle_bullish_pattern'].shift(14)
                )
            )
        )

        # Conditions Sky Jump et Big Down
        dataframe['sky_jump'] = dataframe['red_candle'] & (1 - (dataframe['close'] / dataframe['open'].shift(5)) < -0.5)
        dataframe['big_down'] = dataframe['red_candle'] & (1 - (dataframe['close'] / dataframe['open'].shift(5)) > 0.15)

        # Conditions Recent Fail
        dataframe['recent_fail'] = (
            (metadata['pair'] != 'BTC/USDT') &
            dataframe['is_bearish'] & dataframe['is_bearish'].shift(1) &
            (dataframe['close'].shift(10) > dataframe['close']) &
            (dataframe['buy_ema_2'].shift(10) > dataframe['buy_ema_1'].shift(10)) &
            (dataframe['buy_ema_2'].shift(5) > dataframe['buy_ema_1'].shift(5)) &
            (dataframe['buy_ema_2'].shift(3) > dataframe['buy_ema_1'].shift(3)) &
            (dataframe['buy_ema_2'].shift(1) > dataframe['buy_ema_1'].shift(1)) &
            (dataframe['buy_ema_2'] > dataframe['buy_ema_1'])
        )

        dataframe['is_bearish_fail'] = dataframe['is_bearish'] & dataframe['is_bearish'].shift(1) & dataframe['is_bearish'].shift(2) & (dataframe['big_down'] == False) & (dataframe['big_down'].shift(1) == False)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0
        dataframe['enter_tag'] = ''

        # Vérifiez s'il n'y a pas déjà de trades ouverts pour cette paire
        current_pair = metadata['pair']
        open_trades_for_pair = [trade for trade in Trade.get_open_trades() if trade.pair == current_pair]
        total_open_trades = len(Trade.get_open_trades())

        use_adx = (dataframe['adx'] > 10) | (dataframe['close'] < (dataframe['buy_ema_low'] * 0.95))
        can_enter_long = (True | (metadata['pair'] == 'BTC/USDT') | (total_open_trades < (self.max_open_trades - 1)))

        conditions = {
            'cross_ema': (qtpylib.crossed_above(dataframe['buy_ema_1'], dataframe['buy_ema_2'])) & (can_enter_long) & (dataframe['wait_for_bounce'] == False),
            'dada_price': (metadata['pair'] != 'PEPE/USDT') & (dataframe['close_superior_to_previous_high'] & dataframe['going_up']) & (can_enter_long) & (dataframe['wait_for_bounce'] == False),
            'dada_pattern': (dataframe['dada_pattern']) & (can_enter_long) & (dataframe['wait_for_bounce'] == False),
            # 'rsi': (dataframe['rsi'] < self.rsi_entry_long.value) & (dataframe['rsi_gra'] > 0),
            'big_down': (dataframe['big_down']) & (can_enter_long),
            'bull_engulf': (dataframe['candle_bullish_pattern'] & dataframe['bull_engulfing']) & (can_enter_long) & (dataframe['wait_for_bounce'] == False),
            'hammer': (dataframe['candle_bullish_pattern'] & dataframe['hammer']) & (can_enter_long) & (dataframe['wait_for_bounce'] == False),
            'inv_hammer': (dataframe['candle_bullish_pattern'] & dataframe['inv_hammer']) & (can_enter_long) & (dataframe['wait_for_bounce'] == False),
            'rising_3': (dataframe['candle_bullish_pattern'] & dataframe['rising_3']) & (can_enter_long) & (dataframe['wait_for_bounce'] == False),
            'three_soldiers': (dataframe['candle_bullish_pattern'] & dataframe['three_white_soldiers']) & (can_enter_long) & (dataframe['wait_for_bounce'] == False),
            'morning_star': (dataframe['candle_bullish_pattern'] & dataframe['morning_star']) & (can_enter_long) & (dataframe['wait_for_bounce'] == False),
            'bull_harami': (dataframe['candle_bullish_pattern'] & dataframe['bull_harami']) & (can_enter_long) & (dataframe['wait_for_bounce'] == False),
            # 'tweezer_bottom': (dataframe['candle_bullish_pattern'] & dataframe['tweezer_bottom']) & (can_enter_long) & (dataframe['wait_for_bounce'] == False),
            'dada_back_on_track': (dataframe['dada_back_on_track']) & (can_enter_long & (dataframe['wait_for_bounce'] == False))
        }

        # Appliquer les conditions et définir les tags
        for tag, condition in conditions.items():
            dataframe.loc[condition, 'enter_long'] = 1
            dataframe.loc[condition & (dataframe['enter_tag'] == ''), 'enter_tag'] = tag
        
        # print(dataframe['rsi_gra'].tail(30))

         # Debug print
        # entries_with_bull_engulfing = dataframe[dataframe['hammer']]
        # for index, row in entries_with_bull_engulfing.iterrows():
        #     print(f"hammer detected at index {index}, date {row.name}!")

        # print(dataframe['smooth_k'].tail(10))

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_tag'] = ''

        accept_bearish_pattern = False & (metadata['pair'] == 'BTC/USDT')
        recently_crossed = (dataframe['buy_ema_2'] > dataframe['buy_ema_1']) & ((qtpylib.crossed_above(dataframe['buy_ema_2'].shift(1), dataframe['buy_ema_1'].shift(1))))

        conditions = {
            # 'is_bearish_fail': dataframe['is_bearish_fail'],
            'three_black_crows': accept_bearish_pattern & dataframe['three_black_crows'],
            'shooting_star': accept_bearish_pattern & dataframe['shooting_star'],
            'falling_three': accept_bearish_pattern & dataframe['falling_three'],
            'hanging_man': accept_bearish_pattern & dataframe['hanging_man'],
            # 'rsi': (dataframe['rsi'] > self.rsi_exit_long.value) & (dataframe['rsi_gra'] < 0),
            'sky_jump': dataframe['sky_jump'],
            # 'recent_fail': dataframe['recent_fail'] & (dataframe['enter_long'] == 0) & (dataframe['enter_long'].shift(1) == 0) & (dataframe['enter_long'].shift(2) == 0),
            # 'all_down': dataframe['all_down'],
            'cross_ema':  (qtpylib.crossed_above(dataframe['buy_ema_2'], dataframe['buy_ema_1'])),
            'recently_crossed': recently_crossed,
        }

        # Appliquer les conditions et définir les tags
        for tag, condition in conditions.items():
            dataframe.loc[condition, 'exit_long'] = 1
            dataframe.loc[condition & (dataframe['exit_tag'] == ''), 'exit_tag'] = tag

        return dataframe
    
    def stochastic_oscillator(self, dataframe, length):
        """
        Calcul l'oscillateur stochastique à partir des séries de hauts, de bas et de clôtures.
        Arguments:
        dataframe : DataFrame contenant les colonnes 'high', 'low', et 'close'.
        length : Int, nombre de périodes pour le calcul du stochastique.
        
        Retourne:
        Series contenant les valeurs stochastiques.
        """
        low_min = dataframe['low'].rolling(window=length).min()
        high_max = dataframe['high'].rolling(window=length).max()
        close = dataframe['close']
        
        # Calculer le %K
        K = 100 * ((close - low_min) / (high_max - low_min))
        
        return K