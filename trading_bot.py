import os
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
load_dotenv()

# Ambil env dari GitHub Secrets
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

class IndonesiaStockTradingBot:
    def __init__(self):
        # Daftar saham blue chip dan populer Indonesia (sama seperti scanner.py)
        self.stock_list = [
            # Perbankan (Banks)
            'BBCA.JK', 'BBRI.JK', 'BMRI.JK', 'BNLI.JK',  # BBCA, BBRI, BMRI, Permata Bank
            # Konsumer (Consumer)
            'ASII.JK', 'UNVR.JK', 'ICBP.JK', 'GOTO.JK',  # Astra, Unilever, Indofood, GoTo
            # Telekomunikasi (Telecom)
            'TLKM.JK', 'EXCL.JK', 'ISAT.JK',  # Telkom, XL Axiata, Indosat
            # Pertambangan & Energi (Mining & Energy)
            'ADRO.JK', 'PTBA.JK', 'ITMG.JK', 'BYAN.JK', 'BREN.JK', 'MBMA.JK',  # Adaro, Bukit Asam, Indo Tambang, Bayan Resources, Barito Renewables, Merdeka Battery
            # Infrastruktur (Infrastructure)
            'JSMR.JK', # 'WSKT.JK', 'WIKA.JK',  # Jasa Marga, Waskita, Wijaya Karya
            # Rokok (Tobacco)
            'GGRM.JK', 'HMSP.JK', 'WIIM.JK',  # Gudang Garam, HM Sampoerna, WIKA International
            # Semen (Cement)
            'SMGR.JK', 'INTP.JK',  # Semen Indonesia, Indocement
            # Peternakan & Agribisnis (Agri & Livestock)
            'CPIN.JK', 'JPFA.JK', 'JARR.JK',  # Charoen Pokphand, Japfa, Jhonlin Agro Raya
            # Media, Teknologi & Properti (Media, Tech & Property)
            'MNCN.JK', 'SCMA.JK', 'EMTK.JK', 'COIN.JK', 'PANI.JK',  # MNCN, SCMA, Elang Mahkota, Indokripto, Pantai Indah Kapuk
            # Logam (Metals)
            'ANTM.JK', 'TINS.JK', 'MDKA.JK'   # Aneka Tambang, Timah, Merdeka Copper
            # Personal preference
            'CDIA.JK', 'BRMS.JK', 'BRPT.JK', 'PGAS.JK', 'CUAN.JK', 'BREN.JK', 'PTRO.JK', 'GZCO.JK'  # Chandra Asri, Baramulti, Barito Pacific, Perusahaan Gas Negara, Cuanza, Barito Renewables, Petrosea
        ]

    def get_stock_data(self, symbol, period='1y'):
        """Mengambil data saham dari Yahoo Finance"""
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period=period, interval="1d")
            if len(hist) < 100:  # Minimal data untuk kalkulasi indikator
                return None
            return hist
        except Exception as e:
            print(f"Error fetching {symbol}: {str(e)}")
            return None

    def calculate_rsi(self, prices, window=14):
        """Menghitung RSI (Relative Strength Index)"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Menghitung MACD"""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal).mean()
        return macd, signal_line

    def calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """Menghitung Bollinger Bands"""
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, rolling_mean, lower_band

    def calculate_atr(self, high, low, close, window=14):
        """Menghitung Average True Range (ATR) untuk stop loss"""
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        return atr

    def determine_trend(self, data):
        """Menentukan tren bullish atau bearish"""
        close = data['Close']
        ma_50 = close.rolling(window=50).mean()
        ma_200 = close.rolling(window=200).mean()

        if ma_50.iloc[-1] > ma_200.iloc[-1] and close.iloc[-1] > ma_50.iloc[-1]:
            return 'bullish'
        elif ma_50.iloc[-1] < ma_200.iloc[-1] and close.iloc[-1] < ma_50.iloc[-1]:
            return 'bearish'
        else:
            return 'neutral'

    def get_buy_signal(self, data):
        """Mendapatkan sinyal beli berdasarkan indikator"""
        close = data['Close']
        rsi = self.calculate_rsi(close)
        macd, macd_signal = self.calculate_macd(close)
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close)

        current_price = close.iloc[-1]
        current_rsi = rsi.iloc[-1]
        current_macd = macd.iloc[-1]
        current_macd_signal = macd_signal.iloc[-1]

        # Kondisi untuk beli: RSI oversold, MACD bullish crossover, price near lower BB
        buy_conditions = [
            current_rsi < 30,  # Oversold
            current_macd > current_macd_signal,  # MACD bullish
            current_price <= bb_lower.iloc[-1] * 1.02  # Near lower band
        ]

        if sum(buy_conditions) >= 2:  # Minimal 2 kondisi terpenuhi
            return True, current_price
        return False, None

    def calculate_tp_sl(self, buy_price, trend, atr, volatility_factor=1.5):
        """Menghitung take profit dan stop loss"""
        # Untuk pasar Indonesia yang tidak support short selling retail,
        # semua posisi dianggap long position
        if trend == 'bullish':
            # Long position - target naik
            stop_loss = buy_price - (atr * volatility_factor)  # SL di bawah ATR
            take_profit = buy_price + (atr * volatility_factor * 2)  # TP 2x ATR
        elif trend == 'bearish':
            # Bearish tapi tetap long position (beli di harga rendah, target bounce)
            # Gunakan TP/SL lebih konservatif
            stop_loss = buy_price - (atr * volatility_factor)  # SL di bawah
            take_profit = buy_price + (atr * volatility_factor * 1.5)  # TP lebih konservatif
        else:
            # Neutral, gunakan fixed percentage
            stop_loss = buy_price * 0.95  # 5% SL
            take_profit = buy_price * 1.10  # 10% TP

        return take_profit, stop_loss

    def analyze_stock(self, symbol):
        """Menganalisis saham untuk rekomendasi trading"""
        data = self.get_stock_data(symbol)
        if data is None:
            return None

        trend = self.determine_trend(data)
        buy_signal, buy_price = self.get_buy_signal(data)

        if not buy_signal:
            return None

        # Hitung ATR untuk TP/SL
        atr = self.calculate_atr(data['High'], data['Low'], data['Close'])
        current_atr = atr.iloc[-1]

        take_profit, stop_loss = self.calculate_tp_sl(buy_price, trend, current_atr)

        return {
            'symbol': symbol,
            'company': symbol.replace('.JK', ''),
            'trend': trend,
            'buy_price': buy_price,
            'take_profit': take_profit,
            'stop_loss': stop_loss,
            'atr': current_atr,
            'potential_profit': (take_profit - buy_price) / buy_price * 100,
            'risk': (buy_price - stop_loss) / buy_price * 100
        }

    def run_analysis(self):
        """Menjalankan analisis untuk semua saham"""
        results = []

        print("Memulai analisis trading bot...")
        print("=" * 50)

        for symbol in self.stock_list:
            print(f"Analyzing {symbol}...")

            analysis = self.analyze_stock(symbol)
            if analysis:
                results.append(analysis)

            time.sleep(0.1)  # Rate limiting

        # Sort by potential profit
        results = sorted(results, key=lambda x: x['potential_profit'], reverse=True)

        return results

    def display_results(self, results):
        """Menampilkan hasil analisis"""
        if not results:
            print("Tidak ada rekomendasi trading hari ini.")
            return

        print("\n" + "=" * 80)
        print("HASIL ANALISIS TRADING BOT")
        print("=" * 80)

        for i, stock in enumerate(results, 1):
            print(f"\n{i}. {stock['symbol']} ({stock['company']})")
            print(f"   Tren: {stock['trend'].capitalize()}")
            print(f"   Harga Beli: Rp {stock['buy_price']:,.0f}")
            print(f"   Take Profit: Rp {stock['take_profit']:,.0f} (+{stock['potential_profit']:.1f}%)")
            print(f"   Stop Loss: Rp {stock['stop_loss']:,.0f} (-{stock['risk']:.1f}%)")
            print(f"   ATR: {stock['atr']:.2f}")

    def compose_message(self, results):
        """Menyusun pesan untuk dikirim ke Telegram"""
        if not results:
            return "⚠️ Tidak ada rekomendasi trading hari ini."

        message = "📊 *Rekomendasi Trading Bot Indonesia Stocks*\n\n"
        for stock in results[:5]:  # Top 5 saja
            message += (
                f"• `{stock['symbol']}` ({stock['company']}) | Tren: *{stock['trend'].capitalize()}*\n"
                f"   - Beli: *Rp {stock['buy_price']:,.0f}*\n"
                f"   - TP: *Rp {stock['take_profit']:,.0f}* (+{stock['potential_profit']:.1f}%)\n"
                f"   - SL: *Rp {stock['stop_loss']:,.0f}* (-{stock['risk']:.1f}%)\n\n"
            )
        message += "⚠️ *Disclaimer:* Ini bukan saran keuangan. Lakukan riset sendiri. Gunakan stop loss!"
        return message.strip()

def main():
    # Inisialisasi bot
    bot = IndonesiaStockTradingBot()

    # Jalankan analisis
    results = bot.run_analysis()

    # Tampilkan hasil
    bot.display_results(results)

    # Kirim ke Telegram
    message = bot.compose_message(results)
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }

    try:
        res = requests.post(url, data=payload)
        print("✔️ Terkirim ke Telegram!" if res.status_code == 200 else f"❌ Gagal: {res.text}")
    except Exception as e:
        print(f"Error sending to Telegram: {e}")

if __name__ == "__main__":
    main()