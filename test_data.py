import yfinance as yf
print("Testing yfinance...")
ticker = yf.Ticker("GC=F") # Gold Futures often work better than XAUUSD in some contexts, but let's try standard first
df = ticker.history(period="1mo", interval="15m")
print(f"GC=F 1mo 15m: {len(df)} rows")

ticker = yf.Ticker("XAUUSD=X") # Yahoo finance symbol for Gold
df = ticker.history(period="1mo", interval="15m")
print(f"XAUUSD=X 1mo 15m: {len(df)} rows")
