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

class IndonesiaStockScreener:
  def __init__(self):
    # Daftar saham blue chip dan populer Indonesia (sample)
    self.stock_list = [
      'BBCA.JK', 'BBRI.JK', 'BMRI.JK', 'BBNI.JK', 'BDMN.JK',  # Perbankan
      'ASII.JK', 'UNVR.JK', 'INDF.JK', 'ICBP.JK', 'KLBF.JK', 'NASI.JK',  # Konsumer
      'TLKM.JK', 'EXCL.JK', 'ISAT.JK',  # Telekomunikasi
      'ADRO.JK', 'PTBA.JK', 'ITMG.JK',  # Pertambangan
      'JSMR.JK', 'WSKT.JK', 'WIKA.JK',  # Infrastruktur
      'GGRM.JK', 'HMSP.JK', 'WIIM.JK', 'ITIC.JK',  # Rokok
      'SMGR.JK', 'INTP.JK',  # Semen
      'CPIN.JK', 'JPFA.JK',  # Peternakan
      'MNCN.JK', 'SCMA.JK', 'EMTK.JK',  # Media & Teknologi
      'ANTM.JK', 'TINS.JK', 'MDKA.JK'   # Logam
    ]

  def get_stock_data(self, symbol, period='3mo'):
    """Mengambil data saham dari Yahoo Finance"""
    try:
      stock = yf.Ticker(symbol)
      # hist = stock.download(period=period, interval='1d', progress=False)
      hist = stock.history(period, interval="1d")
      if len(hist) < 20:  # Minimal data untuk kalkulasi indikator
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
    histogram = macd - signal_line
    return macd, signal_line, histogram

  def calculate_bollinger_bands(self, prices, window=20, num_std=2):
    """Menghitung Bollinger Bands"""
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, rolling_mean, lower_band

  def calculate_stochastic(self, high, low, close, k_window=14, d_window=3):
    """Menghitung Stochastic Oscillator"""
    lowest_low = low.rolling(window=k_window).min()
    highest_high = high.rolling(window=k_window).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_window).mean()
    return k_percent, d_percent

  def check_volume_spike(self, volume, window=20, threshold=1.5):
    """Mengecek apakah ada lonjakan volume"""
    avg_volume = volume.rolling(window=window).mean()
    current_volume = volume.iloc[-1]
    avg_volume_recent = avg_volume.iloc[-1]

    if avg_volume_recent == 0:
      return False

    return current_volume > (avg_volume_recent * threshold)

  def swing_trading_criteria(self, data):
    """Kriteria untuk swing trading"""
    close = data['Close']
    high = data['High']
    low = data['Low']
    volume = data['Volume']

    # Hitung indikator
    rsi = self.calculate_rsi(close)
    macd, macd_signal, macd_hist = self.calculate_macd(close)
    bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close)
    stoch_k, stoch_d = self.calculate_stochastic(high, low, close)

    # Moving averages
    ma_20 = close.rolling(window=20).mean()
    ma_50 = close.rolling(window=50).mean()

    # Kondisi terkini
    current_price = close.iloc[-1]
    current_rsi = rsi.iloc[-1]
    current_macd = macd.iloc[-1]
    current_macd_signal = macd_signal.iloc[-1]
    current_stoch_k = stoch_k.iloc[-1]
    current_stoch_d = stoch_d.iloc[-1]

    # Price action
    price_above_ma20 = current_price > ma_20.iloc[-1]
    ma20_above_ma50 = ma_20.iloc[-1] > ma_50.iloc[-1] if not np.isnan(
      ma_50.iloc[-1]) else False

    # Volume check
    volume_spike = self.check_volume_spike(volume)

    # Bollinger Bands position
    bb_position = "middle"
    if current_price > bb_upper.iloc[-1]:
      bb_position = "above_upper"
    elif current_price < bb_lower.iloc[-1]:
      bb_position = "below_lower"

    # Swing Trading Signals
    signals = {
      'bullish_signals': 0,
      'bearish_signals': 0,
      'details': []
    }

    # Bullish signals
    if current_rsi < 70 and current_rsi > 30:  # RSI di zona netral
      if current_rsi > 50:
        signals['bullish_signals'] += 1
        signals['details'].append("RSI bullish (>50)")

    if current_macd > current_macd_signal:  # MACD di atas signal line
      signals['bullish_signals'] += 1
      signals['details'].append("MACD bullish crossover")

    if price_above_ma20 and ma20_above_ma50:  # Trend bullish
      signals['bullish_signals'] += 1
      signals['details'].append("Price above MA20 & MA20>MA50")

    if current_stoch_k > current_stoch_d and current_stoch_k < 80:  # Stochastic bullish tapi belum overbought
      signals['bullish_signals'] += 1
      signals['details'].append("Stochastic bullish")

    if volume_spike:
      signals['bullish_signals'] += 1
      signals['details'].append("Volume spike detected")

    # Bearish signals
    if current_rsi > 70:  # Overbought
      signals['bearish_signals'] += 1
      signals['details'].append("RSI overbought (>70)")

    if current_rsi < 30:  # Oversold (bisa bullish untuk swing)
      signals['details'].append("RSI oversold (<30) - potential reversal")

    if current_macd < current_macd_signal:  # MACD di bawah signal
      signals['bearish_signals'] += 1
      signals['details'].append("MACD bearish")

    if not price_above_ma20:
      signals['bearish_signals'] += 1
      signals['details'].append("Price below MA20")

    # Calculate score
    net_score = signals['bullish_signals'] - signals['bearish_signals']

    return {
      'price': current_price,
      'rsi': current_rsi,
      'macd': current_macd,
      'macd_signal': current_macd_signal,
      'stoch_k': current_stoch_k,
      'stoch_d': current_stoch_d,
      'ma_20': ma_20.iloc[-1],
      'ma_50': ma_50.iloc[-1] if not np.isnan(ma_50.iloc[-1]) else None,
      'bb_position': bb_position,
      'volume_spike': volume_spike,
      'bullish_signals': signals['bullish_signals'],
      'bearish_signals': signals['bearish_signals'],
      'net_score': net_score,
      'signal_details': signals['details']
    }

  def screen_stocks(self, min_score=2):
    """Screen saham berdasarkan kriteria swing trading"""
    results = []

    print("Memulai screening saham untuk swing trading...")
    print("=" * 50)

    for symbol in self.stock_list:
      print(f"Analyzing {symbol}...")

      data = self.get_stock_data(symbol)
      if data is None or len(data) < 50:
        continue

      try:
        analysis = self.swing_trading_criteria(data)

        if analysis['net_score'] >= min_score:
          results.append({
            'symbol': symbol,
            'company': symbol.replace('.JK', ''),
            **analysis
          })

      except Exception as e:
        print(f"Error analyzing {symbol}: {str(e)}")
        continue

      time.sleep(0.1)  # Rate limiting

    # Sort by net score
    results = sorted(results, key=lambda x: x['net_score'], reverse=True)

    return results

  def display_results(self, results):
    """Menampilkan hasil screening"""
    if not results:
      print("Tidak ada saham yang memenuhi kriteria screening.")
      return

    print("\n" + "=" * 80)
    print("HASIL SCREENING SAHAM SWING TRADING")
    print("=" * 80)

    for i, stock in enumerate(results, 1):
      print(f"\n{i}. {stock['symbol']} ({stock['company']})")
      print(f"   Harga: Rp {stock['price']:,.0f}")
      print(f"   Net Score: {stock['net_score']} (Bullish: {stock['bullish_signals']}, Bearish: {stock['bearish_signals']})")
      print(f"   RSI: {stock['rsi']:.1f}")
      print(f"   MACD: {stock['macd']:.3f} (Signal: {stock['macd_signal']:.3f})")
      print(f"   Stochastic K: {stock['stoch_k']:.1f}, D: {stock['stoch_d']:.1f}")
      print(f"   MA20: Rp {stock['ma_20']:,.0f}")
      if stock['ma_50']:
        print(f"   MA50: Rp {stock['ma_50']:,.0f}")
      print(f"   Bollinger Position: {stock['bb_position']}")
      print(f"   Volume Spike: {'Ya' if stock['volume_spike'] else 'Tidak'}")
      print(f"   Sinyal: {', '.join(stock['signal_details'])}")

  def compose_message(self, results):
    """Menyusun pesan untuk dikirim ke Telegram"""
    if not results:
      return "⚠️ Tidak ada saham yang lolos screening swing trading hari ini."

    message = "📊 <b>Hasil Screening Top 5 Saham Swing Trading</b>\n\n"
    for stock in results[:5]:  # Top 5 saja
      message += (
        f"• <code>{stock['symbol']}</code> ({stock['company']}) | Harga: <b>Rp {stock['price']:,.0f}</b> | "
        f"Net Score: <b>{stock['net_score']}</b> (Bullish: {stock['bullish_signals']}, Bearish: {stock['bearish_signals']})\n"
        f"   - RSI: {stock['rsi']:.1f}, MACD: {stock['macd']:.3f} (Signal: {stock['macd_signal']:.3f})\n"
        f"   - Stochastic K: {stock['stoch_k']:.1f}, D: {stock['stoch_d']:.1f}\n"
        f"   - MA20: Rp {stock['ma_20']:,.0f}" + (
            f", MA50: Rp {stock['ma_50']:,.0f}" if stock['ma_50'] else "") + "\n"
        f"   - Bollinger Position: {stock['bb_position']}, Volume Spike: {'Ya' if stock['volume_spike'] else 'Tidak'}\n"
        f"   - Sinyal: {', '.join(stock['signal_details'])}\n\n"
      )
    return message.strip()

  def export_to_csv(self, results, filename=None):
    """Export hasil ke CSV"""
    if not results:
      return

    if filename is None:
      filename = f"swing_trading_screening_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"

    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"\nHasil screening telah disimpan ke: {filename}")

    return filename


def main():
  # Inisialisasi screener
  screener = IndonesiaStockScreener()

  # Jalankan screening dengan minimum score 2
  results = screener.screen_stocks(min_score=1)  # Lowered for demo

  # Tampilkan hasil
  screener.display_results(results)

  # Export ke CSV
  filename = None
  if results:
    filename = screener.export_to_csv(results)

  # Kirim ke Telegram
  message = screener.compose_message(results)
  url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
  payload = {
    "chat_id": CHAT_ID,
    "text": message,
    "parse_mode": "HTML"
  }
  
  res = requests.post(url, data=payload)
  print("✔️ Terkirim ke Telegram!" if res.status_code == 200 else f"❌ Gagal: {res.text}")

  if filename:
    url = f"https://api.telegram.org/bot{TOKEN}/sendDocument"
    payload = {
      "chat_id": CHAT_ID,
      "caption": "📈 Hasil lengkap screening swing trading.",
    }
    files = {
      'document': open(filename, 'rb')
    }
    res = requests.post(url, data=payload, files=files)
    print("✔️ File CSV terkirim ke Telegram!" if res.status_code == 200 else f"❌ Gagal kirim file: {res.text}")

  # Tips trading
  print("\n" + "=" * 80)
  print("TIPS SWING TRADING:")
  print("=" * 80)
  print("1. Selalu gunakan stop loss (biasanya 3-5% dari entry point)")
  print("2. Target profit swing trading biasanya 10-20%")
  print("3. Perhatikan support dan resistance levels")
  print("4. Kombinasikan dengan analisis fundamental")
  print("5. Gunakan proper position sizing (maksimal 5% dari portfolio per saham)")
  print("6. Perhatikan kalender ekonomi dan berita perusahaan")
  print("7. Best time frame untuk swing trading: daily dan 4H charts\n")
  
  # Clean up
  if os.path.exists(filename):
    try:
      os.remove(filename)
      print('CSV File sudah dihapus')
    except OSError as err:
      print('Error saat menghapus file: ', err)

if __name__ == "__main__":
  main()
