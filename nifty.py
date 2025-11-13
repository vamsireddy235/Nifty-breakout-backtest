import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import sys
import numpy as np

# === CONFIG ===
symbol = input("Enter Yahoo Finance symbol (default ^NSEI or ^NSEBANK): ").strip() or "^NSEI"
interval = "5m"
max_target_pts = 30

print(f"\n📥 Fetching 60 days of {interval} data for {symbol} ...")

# === Download 60 days of intraday data ===
try:
    data = yf.download(tickers=symbol, interval=interval, period="60d", prepost=False, progress=False)
except Exception as e:
    print(f"⚠️ Error fetching data: {e}")
    sys.exit()

if data.empty:
    print(f"⚠️ No data received for {symbol}.")
    sys.exit()

# === Prepare Data ===
data.reset_index(inplace=True)
data.rename(columns={'Datetime': 'datetime'}, inplace=True)
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

# Localize to India timezone
try:
    data.index = data.index.tz_convert("Asia/Kolkata")
except:
    data.index = data.index.tz_localize("Asia/Kolkata")

# === Backtest Setup ===
results = []
unique_days = sorted(list(set([ts.date() for ts in data.index])))

print(f"\n📊 Running backtest for {len(unique_days)} trading days...\n")

for today in unique_days:
    session_start = pd.Timestamp(f"{today} 09:15:00", tz="Asia/Kolkata")
    session_end   = pd.Timestamp(f"{today} 09:45:00", tz="Asia/Kolkata")

    # Select today’s data
    df_day = data[(data.index.date == today)]
    if df_day.empty:
        continue

    first_candle = df_day[(df_day.index >= session_start) & (df_day.index <= session_end)]
    if first_candle.empty:
        continue

    # === Identify 30-min candle details ===
    open_30m = first_candle['Open'].iloc[0].item()
    close_30m = first_candle['Close'].iloc[-1].item()
    high_30m = first_candle['High'].max().item()
    low_30m = first_candle['Low'].min().item()

    # Determine candle color and stop-loss reference
    if close_30m > open_30m:
        candle_color = "GREEN"
        candle_stoploss = low_30m
    else:
        candle_color = "RED"
        candle_stoploss = high_30m

    post_session = df_day[df_day.index > session_end]

    trade_taken = False
    direction = None
    entry_price = None
    entry_time = None
    stop_loss = None
    target = None
    gain = 0.0
    exit_time = None

    for time, row in post_session.iterrows():
        close = row['Close'].item()
        high = row['High'].item()
        low = row['Low'].item()

        if not trade_taken:
            if close > high_30m:
                direction = "BUY"
                entry_price = close
                entry_time = time
                stop_loss = candle_stoploss
                raw_target = entry_price + (entry_price - stop_loss)
                target = min(raw_target, entry_price + max_target_pts)
                trade_taken = True
            elif close < low_30m:
                direction = "SELL"
                entry_price = close
                entry_time = time
                stop_loss = candle_stoploss
                raw_target = entry_price - (stop_loss - entry_price)
                target = max(raw_target, entry_price - max_target_pts)
                trade_taken = True
        else:
            if direction == "BUY":
                if low <= stop_loss:
                    gain = stop_loss - entry_price
                    exit_time = time
                    break
                elif high >= target:
                    gain = target - entry_price
                    exit_time = time
                    break
            elif direction == "SELL":
                if high >= stop_loss:
                    gain = entry_price - stop_loss
                    exit_time = time
                    break
                elif low <= target:
                    gain = entry_price - target
                    exit_time = time
                    break

    if trade_taken and gain == 0.0:
        # End of day exit
        last_close = post_session['Close'].iloc[-1].item()
        gain = (last_close - entry_price) if direction == "BUY" else (entry_price - last_close)
        exit_time = post_session.index[-1]

    gain_capped = max(-max_target_pts, min(gain, max_target_pts))
    weekday = pd.Timestamp(today).strftime("%A")  # NEW: weekday name

    results.append({
        "Date": today,
        "Weekday": weekday,                       # NEW COLUMN
        "Candle Color": candle_color,
        "Direction": direction if direction else "No Trade",
        "Entry Time": entry_time.time() if entry_time is not None else None,
        "Exit Time": exit_time.time() if exit_time is not None else None,
        "Gain (pts)": round(gain_capped, 2)
    })

# === Show Results ===
df_results = pd.DataFrame(results)
if df_results.empty:
    print("⚠️ No trades found in 60-day period.")
    sys.exit()

df_results["Cumulative Gain"] = df_results["Gain (pts)"].cumsum()

print("\n📅 60-Day Breakout Backtest Results:")
print(df_results.to_string(index=False))

total_gain = df_results["Gain (pts)"].sum()
avg_gain = df_results["Gain (pts)"].mean()
win_days = (df_results["Gain (pts)"] > 0).sum()
loss_days = (df_results["Gain (pts)"] < 0).sum()

print(f"\n📈 Summary for {symbol}:")
print(f"Total Days Tested : {len(df_results)}")
print(f"Winning Days      : {win_days}")
print(f"Losing Days       : {loss_days}")
print(f"Average Gain/Day  : {avg_gain:.2f} pts")
print(f"Total Net Gain    : {total_gain:.2f} pts")


# === Weekday-wise Wins, Losses, and Average Gain ===
weekday_summary = df_results.groupby("Weekday").agg(
    Winning_Days=("Gain (pts)", lambda x: (x > 0).sum()),
    Losing_Days=("Gain (pts)", lambda x: (x < 0).sum()),
    No_Trade_Days=("Direction", lambda x: (x == "No Trade").sum()),
    Total_Gain=("Gain (pts)", "sum"),
    Average_Gain=("Gain (pts)", "mean")
).reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])

print("\n📅 Weekday-wise Performance:")
print(weekday_summary.to_string())


#visualization

#Bar Chart for Wins, Losses, and No-Trade Days per Weekday

import matplotlib.pyplot as plt

weekday_summary[["Winning_Days", "Losing_Days", "No_Trade_Days"]].plot(
    kind="bar", figsize=(10,6), stacked=True, color=["#5CFF5C", "#FF2E2E", "gray"]
)
plt.title(f"{symbol} - Weekday-wise Win/Loss/No-Trade Days")
plt.ylabel("Number of Days")
plt.xlabel("Weekday")
plt.xticks(rotation=0)
plt.show()


# 4. Grouped Bar Chart: Wins/Losses per Weekday
# Shows weekly patterns in performance
idx = weekday_summary.index
width = 0.3
x = np.arange(len(idx))
plt.figure(figsize=(10, 6))
plt.bar(x - width, weekday_summary['Winning_Days'], width=width, label='Winning Days', color='green')
plt.bar(x,         weekday_summary['Losing_Days'], width=width, label='Losing Days', color='red')
plt.bar(x + width, weekday_summary['No_Trade_Days'], width=width, label='No Trade', color='gray')
plt.xticks(x, idx)
plt.ylabel('Number of Days')
plt.title('Wins/Losses/No Trade by Weekday')
plt.legend()
plt.tight_layout()
plt.show()



#Bar Chart for Total Gain per Weekday

weekday_summary["Total_Gain"].plot(
    kind="bar", figsize=(8,5), color="skyblue"
)
plt.title(f"{symbol} - Total Gain per Weekday")
plt.ylabel("Total Gain (pts)")
plt.xlabel("Weekday")
plt.xticks(rotation=0)
plt.show()


#Line Chart for Average Gain per Weekday
weekday_summary["Average_Gain"].plot(
    kind="line", marker="o", figsize=(8,5), color="blue"
)
plt.title(f"{symbol} - Average Gain per Weekday")
plt.ylabel("Average Gain (pts)")
plt.xlabel("Weekday")
plt.xticks(rotation=0)
plt.grid(True)
plt.show()


import matplotlib.pyplot as plt


# 1. Line Chart: Cumulative Gain by Date
# Shows the long-term equity curve
plt.figure(figsize=(10, 5))
plt.plot(df_results['Date'], df_results['Cumulative Gain'], marker='o')
plt.xlabel('Date')
plt.ylabel('Cumulative Gain (pts)')
plt.title('Cumulative Gain Over Time')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# 3. Pie/Donut Chart: Win, Loss, No Trade
# Displays proportions of trade outcomes
outcomes = [
    (df_results["Gain (pts)"] > 0).sum(),
    (df_results["Gain (pts)"] < 0).sum(),
    (df_results["Direction"] == "No Trade").sum()
]
labels = ['Winning Days', 'Losing Days', 'No Trade']
plt.figure(figsize=(6, 6))
plt.pie(outcomes, labels=labels, autopct='%1.1f%%', startangle=140, wedgeprops=dict(width=0.5))
plt.title('Trade Outcomes Distribution')
plt.show()


import matplotlib.pyplot as plt

# Calculate percentages
weekday_percent = weekday_summary[["Winning_Days", "Losing_Days", "No_Trade_Days"]].div(
    weekday_summary[["Winning_Days", "Losing_Days", "No_Trade_Days"]].sum(axis=1), axis=0
) * 100

# Plot 100% stacked bar chart
weekday_percent.plot(
    kind="bar",
    stacked=True,
    figsize=(10,6),
    color=["green", "red", "gray"]
)

plt.ylabel("Percentage (%)")
plt.xlabel("Weekday")
plt.title(f"{symbol} - Weekday-wise Win/Loss/No-Trade Percentage")
plt.xticks(rotation=0)
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()




import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure 'Date' is datetime and set as index
df_results['Date'] = pd.to_datetime(df_results['Date'])
df_results.set_index('Date', inplace=True)

# Extract year, month, day, and weekday for calendar layout
df_results['Year'] = df_results.index.year
df_results['Month'] = df_results.index.month
df_results['Day'] = df_results.index.day
df_results['Weekday'] = df_results.index.weekday  # Monday=0

# Function to plot calendar heatmap for a particular year-month
def plot_month_heatmap(year, month):
    month_data = df_results[(df_results['Year'] == year) & (df_results['Month'] == month)]
    if month_data.empty:
        print(f"No data for {year}-{month:02d}")
        return

    # Pivot: rows=Day of month, columns=Weekday (Mon=0)
    pivot = month_data.pivot(index='Day', columns='Weekday', values='Gain (pts)')
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap='RdYlGn', center=0, linewidths=0.5, cbar_kws={'label': 'Gain (pts)'})
    plt.title(f'Calendar Heatmap of Daily Gains: {year}-{month:02d}')
    plt.xlabel('Weekday (Mon=0)')
    plt.ylabel('Day of Month')
    plt.show()

# Example: Plot heatmap for the most recent month in data
latest_year = df_results['Year'].max()
latest_month = df_results[df_results['Year'] == latest_year]['Month'].max()
plot_month_heatmap(latest_year, latest_month)














import matplotlib.pyplot as plt

# Calculate winning percentage per weekday
weekday_summary['Winning_Percentage'] = (
    weekday_summary['Winning_Days'] / 
    (weekday_summary['Winning_Days'] + weekday_summary['Losing_Days']) * 100
).round(2)

# Data for donut chart
sizes = weekday_summary['Winning_Percentage']
labels = weekday_summary.index

# Light color palette for weekdays
colors = ['#A8E6CF', '#DCEDC1', '#FFD3B6', '#FFAAA5', '#FF8B94']

plt.figure(figsize=(8,8))
wedges, texts, autotexts = plt.pie(
    sizes,
    labels=None,  # Hide labels on slices for clarity
    autopct='%1.1f%%',
    startangle=90,
    colors=colors,
    wedgeprops=dict(width=0.4, edgecolor='white'),
    pctdistance=0.75,  # % inside donut
    textprops=dict(weight='bold', color='#333333', fontsize=12)
)

# Add weekday labels outside the donut
for i, p in enumerate(wedges):
    angle = (p.theta2 - p.theta1)/2. + p.theta1
    y = 1.2 * np.sin(np.deg2rad(angle))
    x = 1.2 * np.cos(np.deg2rad(angle))
    plt.text(x, y, f"{labels[i]}", ha='center', va='center', fontsize=12, weight='bold', color='#555555')

plt.title(f"{symbol} - Winning Percentage per Weekday", fontsize=16, fontweight='bold')
plt.axis('equal')  # Equal aspect ratio ensures donut is circular
plt.show()


#donut chart for losing percentage per weekday
import matplotlib.pyplot as plt
import numpy as np

# Calculate losing percentage per weekday
weekday_summary['Losing_Percentage'] = (
    weekday_summary['Losing_Days'] / 
    (weekday_summary['Winning_Days'] + weekday_summary['Losing_Days']) * 100
).round(2)

# Data for donut chart
sizes = weekday_summary['Losing_Percentage']
labels = weekday_summary.index

# Light color palette for losing slices
colors = ['#FFCDD2', '#EF9A9A', '#E57373', '#EF5350', '#F44336']

plt.figure(figsize=(8,8))
wedges, texts, autotexts = plt.pie(
    sizes,
    labels=None,  # Hide labels on slices
    autopct='%1.1f%%',
    startangle=90,
    colors=colors,
    wedgeprops=dict(width=0.4, edgecolor='white'),
    pctdistance=0.75,  # % inside donut
    textprops=dict(weight='bold', color='#333333', fontsize=12)
)

# Add weekday labels outside the donut
for i, p in enumerate(wedges):
    angle = (p.theta2 - p.theta1)/2. + p.theta1
    y = 1.2 * np.sin(np.deg2rad(angle))
    x = 1.2 * np.cos(np.deg2rad(angle))
    plt.text(x, y, f"{labels[i]}", ha='center', va='center', fontsize=12, weight='bold', color='#555555')

plt.title(f"{symbol} - Losing Percentage per Weekday", fontsize=16, fontweight='bold')
plt.axis('equal')  # Keep donut circular
plt.show()


import matplotlib.pyplot as plt
import pandas as pd

# Ensure winning and losing percentages sum to 100
weekday_summary['Total_Trades'] = weekday_summary['Winning_Days'] + weekday_summary['Losing_Days']
weekday_summary['Winning_Percentage'] = (weekday_summary['Winning_Days'] / weekday_summary['Total_Trades'] * 100).round(2)
weekday_summary['Losing_Percentage'] = (weekday_summary['Losing_Days'] / weekday_summary['Total_Trades'] * 100).round(2)

# Plot 100% stacked bar chart
fig, ax = plt.subplots(figsize=(10,6))

ax.bar(weekday_summary.index, weekday_summary['Winning_Percentage'], label='Winning %', color='#79f299')
ax.bar(weekday_summary.index, weekday_summary['Losing_Percentage'], bottom=weekday_summary['Winning_Percentage'], label='Losing %', color='#f75f54')

# Add data labels
for i, row in weekday_summary.iterrows():
    ax.text(i, row['Winning_Percentage']/2, f"{row['Winning_Percentage']}%", ha='center', va='center', color='black', fontweight='bold')
    ax.text(i, row['Winning_Percentage'] + row['Losing_Percentage']/2, f"{row['Losing_Percentage']}%", ha='center', va='center', color='black', fontweight='bold')

ax.set_ylabel('Percentage (%)')
ax.set_title(f"{symbol} - Winning vs Losing Percentage per Weekday", fontsize=14, fontweight='bold')
ax.legend(loc='upper right')
plt.xticks(rotation=0)
plt.ylim(0, 100)
plt.tight_layout()
plt.show()


# === Final Backtest Summary Table Visualization (3 Decimal Precision) ===
import matplotlib.pyplot as plt
import pandas as pd

# --- Replace emoji to avoid font warnings ---
symbol_display = symbol  # You can append 📊 if your font supports it

# --- Summary Data ---
summary_data = {
    "Metric": ["Total Days Tested", "Winning Days", "Losing Days", "Average Gain/Day", "Total Net Gain"],
    "Value": [len(df_results), win_days, loss_days, f"{avg_gain:.3f} pts", f"{total_gain:.3f} pts"]
}
summary_df = pd.DataFrame(summary_data)

# --- Weekday Summary ---
weekday_reset = weekday_summary.reset_index().copy()
weekday_reset = weekday_reset.round(3)  # round numeric values to 3 decimals

# --- Figure Setup ---
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

# --- Title ---
ax.set_title(f"Final Backtest Summary — {symbol_display}", fontsize=16, fontweight='bold', pad=20)

# --- Adjust column widths ---
col_width_summary = [0.5, 0.3]
col_width_weekday = [0.18] * len(weekday_reset.columns)

# --- Summary Table (Top Section) ---
table_summary = ax.table(
    cellText=summary_df.values,
    colLabels=summary_df.columns,
    colWidths=col_width_summary,
    cellLoc='center',
    loc='upper center',
    bbox=[0.05, 0.75, 0.9, 0.2]
)
table_summary.auto_set_font_size(False)
table_summary.set_fontsize(11)

# Header and row styling
for (row, col), cell in table_summary.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#4C72B0')
    else:
        cell.set_facecolor('#EAF2FA')

# --- Weekday Summary Table (Bottom Section) ---
table_weekday = ax.table(
    cellText=weekday_reset.values,
    colLabels=weekday_reset.columns,
    colWidths=col_width_weekday,
    cellLoc='center',
    loc='center',
    bbox=[0.05, 0.05, 0.9, 0.65]
)
table_weekday.auto_set_font_size(False)
table_weekday.set_fontsize(9)

# Alternate row colors & header style
for (row, col), cell in table_weekday.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#2C7BB6')
    elif row % 2 == 0:
        cell.set_facecolor('#F5F5F5')
    else:
        cell.set_facecolor('#FFFFFF')

# --- Layout and Display ---
plt.tight_layout()
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------
# Ensure df_results exists and has data
# ------------------------------
df_results = pd.DataFrame(results)  # results from your backtest loop

if df_results.empty or 'Date' not in df_results.columns:
    print("❌ No trade data found. Cannot plot table!")
else:
    # Ensure 'Date' is datetime
    df_results['Date'] = pd.to_datetime(df_results['Date'])
    
    # Calculate cumulative gain if not already done
    if 'Cumulative Gain' not in df_results.columns:
        df_results['Cumulative Gain'] = df_results['Gain (pts)'].cumsum()
    
    # Round numeric columns for display
    df_results = df_results.round({'Gain (pts)': 3, 'Cumulative Gain': 3})
    
    # Convert times to string for display
    df_results['Entry Time'] = df_results['Entry Time'].astype(str)
    df_results['Exit Time'] = df_results['Exit Time'].astype(str)
    
    # ------------------------------
    # Keep only the latest 30 days
    # ------------------------------
    df_display = df_results.tail(30).copy()
    
    # ------------------------------
    # Plot Latest 30-Day Daily Results Table
    # ------------------------------
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('off')
    ax.set_title(f"Latest 30-Day Breakout Backtest Results — {symbol}", fontsize=16, fontweight='bold', pad=20)
    
    table_daily = ax.table(
        cellText=df_display.values,
        colLabels=df_display.columns,
        cellLoc='center',
        colWidths=[0.09]*len(df_display.columns),
        bbox=[0.01, 0.01, 0.98, 0.95]
    )
    
    table_daily.auto_set_font_size(False)
    table_daily.set_fontsize(9)
    
    # Column indices
    col_candle = df_display.columns.get_loc("Candle Color")
    col_gain = df_display.columns.get_loc("Gain (pts)")
    
    # Alternate row colors & header style
    for (row, col), cell in table_daily.get_celld().items():
        if row == 0:
            # Header
            cell.set_facecolor('#4C72B0')
            cell.set_text_props(weight='bold', color='white')
        else:
            # Alternate row background
            if row % 2 == 0:
                cell.set_facecolor('#F5F5F5')
            else:
                cell.set_facecolor('#FFFFFF')
            
            # Candle Color text
            if col == col_candle:
                if cell.get_text().get_text() == "GREEN":
                    cell.get_text().set_color("green")
                    cell.get_text().set_weight("bold")
                elif cell.get_text().get_text() == "RED":
                    cell.get_text().set_color("red")
                    cell.get_text().set_weight("bold")
            
            # Gain (pts) background
            if col == col_gain:
                try:
                    val = float(cell.get_text().get_text())
                    if val > 0:
                        cell.set_facecolor('#C6EFCE')  # light green
                    elif val < 0:
                        cell.set_facecolor('#FFC7CE')  # light red
                    else:
                        cell.set_facecolor('#FFFFFF')
                except:
                    pass  # ignore non-numeric
                
    plt.tight_layout()
    plt.show()
    
 
# Top 3 Winning Weekdays
top_weekdays = weekday_summary.sort_values("Average_Gain", ascending=False).head(3)
print("🏆 Top 3 Winning Weekdays:")
print(top_weekdays[["Average_Gain", "Total_Gain"]])

# Top 3 Worst Weekdays
worst_weekdays = weekday_summary.sort_values("Average_Gain").head(3)
print("💀 Worst 3 Weekdays:")
print(worst_weekdays[["Average_Gain", "Total_Gain"]])



import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

# --- Ensure top and worst weekdays data ---
top_weekdays_plot = top_weekdays[['Average_Gain', 'Total_Gain']].copy()
top_weekdays_plot['Average_Gain'] = top_weekdays_plot['Average_Gain'].round(3)

worst_weekdays_plot = worst_weekdays[['Average_Gain', 'Total_Gain']].copy()
worst_weekdays_plot['Average_Gain'] = worst_weekdays_plot['Average_Gain'].round(3)

# --- Set font that supports emojis ---
mpl.rcParams['font.family'] = 'Segoe UI Emoji'  # Windows
# mpl.rcParams['font.family'] = 'Apple Color Emoji'  # macOS
# mpl.rcParams['font.family'] = 'Noto Color Emoji'  # Linux

# --- Create figure ---
fig, axes = plt.subplots(2, 1, figsize=(8, 6))
plt.subplots_adjust(hspace=0.6)

# --- Top 3 Winning Weekdays Table ---
axes[0].axis('off')
tbl_top = axes[0].table(
    cellText=top_weekdays_plot.reset_index().values,
    colLabels=top_weekdays_plot.reset_index().columns,
    cellLoc='center',
    loc='center'
)
tbl_top.auto_set_font_size(False)
tbl_top.set_fontsize(12)
tbl_top.scale(1, 1.5)
for (i, j), cell in tbl_top.get_celld().items():
    if i == 0:
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white')
    else:
        cell.set_facecolor('#C6EFCE')

axes[0].set_title("🏆 Top 3 Winning Weekdays", fontsize=14, fontweight='bold', pad=10)

# --- Worst 3 Weekdays Table ---
axes[1].axis('off')
tbl_worst = axes[1].table(
    cellText=worst_weekdays_plot.reset_index().values,
    colLabels=worst_weekdays_plot.reset_index().columns,
    cellLoc='center',
    loc='center'
)
tbl_worst.auto_set_font_size(False)
tbl_worst.set_fontsize(12)
tbl_worst.scale(1, 1.5)
for (i, j), cell in tbl_worst.get_celld().items():
    if i == 0:
        cell.set_facecolor('#F44336')
        cell.set_text_props(weight='bold', color='white')
    else:
        cell.set_facecolor('#FFC7CE')

axes[1].set_title("💀 Worst 3 Weekdays", fontsize=14, fontweight='bold', pad=10)

plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd

# Ensure df_results has datetime index and cumulative gain
df_results['Date'] = pd.to_datetime(df_results['Date'])
df_results.set_index('Date', inplace=True)
if 'Cumulative Gain' not in df_results.columns:
    df_results['Cumulative Gain'] = df_results['Gain (pts)'].cumsum()

# Fetch NSEI intraday data with explicit auto_adjust to avoid warning
nsei = yf.download("^NSEI", interval="5m", period="60d", progress=False, auto_adjust=True)
nsei['Close'] = nsei['Close'].ffill()

# Resample to daily close
nsei_daily = nsei['Close'].resample('D').last()

# Compute cumulative % change without filling method warning
nsei_cum = nsei_daily.pct_change(fill_method=None).cumsum() * 1000  # scale for visibility

# Plot Strategy vs NSEI with solid blue line
plt.figure(figsize=(12,6))
plt.plot(df_results.index, df_results['Cumulative Gain'], marker='o', label='Strategy', color='green')
plt.plot(nsei_cum.index, nsei_cum, marker='o', label='NSEI % Change x1000', color='royalblue')
plt.title(f"{symbol} Strategy vs NSEI Index", fontsize=14, fontweight='bold')
plt.xlabel("Date")
plt.ylabel("Cumulative Gain / Relative Performance")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()




