import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import time

# ---------------- INDEX SYMBOL MAP ----------------
index_symbol_map = {
    "NIFTY 50": "^NSEI",
    "NIFTY Bank": "^NSEBANK",
    "NIFTY Next 50": "^NSMIDCP",
    "NIFTY 100": "^CNX100",
    "NIFTY 200": "^CNX200",
    "NIFTY 500": "^CRSLDX",
    "NIFTY Midcap 50": "^CNXMID50",
    "NIFTY Midcap 100": "NIFTY_MIDCAP_100.NS",
    "NIFTY Smallcap 100": "^CNXSC",
    "NIFTY IT": "^CNXIT",
    "NIFTY Financial Services": "^CNXFIN",
    "NIFTY FMCG": "^CNXFMCG",
    "NIFTY Pharma": "^CNXPHARMA",
    "NIFTY Auto": "^CNXAUTO",
    "NIFTY Metal": "^CNXMETAL",
    "NIFTY Energy": "^CNXENERGY",
    "NIFTY Realty": "^CNXREALTY",
    "NIFTY Private Bank": "NIFTY_PVT_BANK.NS",
    "NIFTY Services": "^CNXSERVICE",
    "NIFTY Consumption": "^CNXCONSUMPTION",
    "NIFTY Infrastructure": "^CNXINFRA",
    "BSE IT": "BSE-IT.BO",
    "BSE Energy": "BSE-ENERGY.BO",
    "BSE FMCG": "BSE-FMCG.BO",
    "BSE Metal": "BSE-METAL.BO",
    "BSE Realty": "BSE-REALTY.BO"
}

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="Intraday Breakout Strategy", layout="wide")
st.title("📈 Intraday Breakout Backtest Strategy")

# ---------------- USER INPUTS ----------------
selected_index = st.selectbox("Select Index:", list(index_symbol_map.keys()))
selected_symbol = index_symbol_map[selected_index]
backtest_days = st.selectbox("Select Backtest Period (days):", [60, 50, 40, 30, 20, 10])

col1, col2, col3 = st.columns(3)
run_backtest = col1.button("🚀 Run Backtest")
show_last_trade = col2.button("📊 Last Trade (All Indexes)")
run_all_backtest = col3.button("📈 Run Backtest (All Indexes)")

# ---------------- LOAD DATA ----------------
@st.cache_data(show_spinner=True)
def load_data(symbol, days):
    df = yf.download(tickers=symbol, interval='5m', period=f'{days}d', prepost=False, progress=False)
    if df.empty:
        return pd.DataFrame()
    df = df.reset_index().rename(columns={'Datetime': 'datetime'})
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    try:
        df.index = df.index.tz_convert("Asia/Kolkata")
    except:
        df.index = df.index.tz_localize("Asia/Kolkata")
    return df

# ---------------- STRATEGY FUNCTION ----------------
def run_strategy(data):
    max_target_pts = 30
    results = []
    unique_days = sorted(list(set([ts.date() for ts in data.index])))

    for today in unique_days:
        session_start = pd.Timestamp(f"{today} 09:15:00", tz="Asia/Kolkata")
        session_end = pd.Timestamp(f"{today} 09:45:00", tz="Asia/Kolkata")

        df_day = data[data.index.date == today].copy()
        if df_day.empty:
            continue

        first_candle = df_day[(df_day.index >= session_start) & (df_day.index <= session_end)].copy()
        if first_candle.empty:
            continue

        open_30m = float(first_candle['Open'].iloc[0])
        close_30m = float(first_candle['Close'].iloc[-1])
        high_30m = float(first_candle['High'].max())
        low_30m = float(first_candle['Low'].min())

        if close_30m > open_30m:
            candle_color = "GREEN"
            candle_stoploss = low_30m
        else:
            candle_color = "RED"
            candle_stoploss = high_30m

        post_session = df_day[df_day.index > session_end].copy()

        trade_taken = False
        direction = None
        entry_price = None
        entry_time = None
        stop_loss = None
        target = None
        gain = 0.0
        exit_time = None

        for time, row in post_session.iterrows():
            close = float(row['Close'])
            high = float(row['High'])
            low = float(row['Low'])

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
            last_close = float(post_session['Close'].iloc[-1])
            gain = (last_close - entry_price) if direction == "BUY" else (entry_price - last_close)
            exit_time = post_session.index[-1]

        gain_capped = max(-max_target_pts, min(gain, max_target_pts))
        weekday = pd.Timestamp(today).strftime("%A")

        results.append({
            "Date": today,
            "Weekday": weekday,
            "Candle Color": candle_color,
            "Direction": direction if direction else "No Trade",
            "Entry Time": entry_time.time() if entry_time is not None else None,
            "Exit Time": exit_time.time() if exit_time is not None else None,
            "Gain (pts)": round(gain_capped, 2)
        })

    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results["Cumulative Gain"] = df_results["Gain (pts)"].cumsum()
    return df_results


# ---------------- RUN BACKTEST FOR ALL INDEXES ----------------
if run_all_backtest:
    st.info(f"Running backtest for all indexes for last {backtest_days} days... ⏳")
    all_summaries = []

    for idx_name, idx_symbol in index_symbol_map.items():
        data = load_data(idx_symbol, backtest_days)
        if data.empty:
            continue
        df_results = run_strategy(data)
        if df_results.empty:
            continue

        total_days = len(df_results)
        winning_days = (df_results["Gain (pts)"] > 0).sum()
        losing_days = (df_results["Gain (pts)"] < 0).sum()
        avg_gain = df_results["Gain (pts)"].mean()
        total_gain = df_results["Gain (pts)"].sum()

        all_summaries.append({
            "Index": idx_name,
            "Total Days": total_days,
            "Winning Days": winning_days,
            "Losing Days": losing_days,
            "Avg Gain/Day": round(avg_gain, 2),
            "Total Gain": round(total_gain, 2)
        })

    if not all_summaries:
        st.error("No data/trades found for any index.")
    else:
        df_summary_all = pd.DataFrame(all_summaries)
        df_summary_all = df_summary_all.sort_values(by="Total Gain", ascending=False)

        # --- Matplotlib Table Visualization ---
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.axis('off')
        ax.set_title(f"📊 Backtest Summary for All Indexes — Last {backtest_days} Days", fontsize=16, fontweight='bold', pad=20)

        table = ax.table(
            cellText=df_summary_all.values,
            colLabels=df_summary_all.columns,
            cellLoc='center',
            colWidths=[0.15]*len(df_summary_all.columns),
            bbox=[0.01, 0.01, 0.98, 0.95]
        )

        table.auto_set_font_size(False)
        table.set_fontsize(11)

        # Color headers and gain column
        col_total_gain = df_summary_all.columns.get_loc("Total Gain")
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor('#4C72B0')
                cell.set_text_props(weight='bold', color='white')
            else:
                if row % 2 == 0:
                    cell.set_facecolor('#F5F5F5')
                else:
                    cell.set_facecolor('#FFFFFF')

                if col == col_total_gain:
                    try:
                        val = float(cell.get_text().get_text())
                        if val > 0:
                            cell.set_facecolor('#C6EFCE')
                        elif val < 0:
                            cell.set_facecolor('#FFC7CE')
                    except:
                        pass

        plt.tight_layout()
        st.pyplot(fig)
        
        # ----------- NEW FEATURE: Compare Top 5 Indexes ------------
        st.subheader(f"🏆 Top 5 Index Performance Comparison — Last {backtest_days} Days")

        progress = st.progress(0)
        all_comparison = []

        total_indexes = len(index_symbol_map)
        for i, (idx_name, idx_symbol) in enumerate(index_symbol_map.items()):
            progress.progress((i + 1) / total_indexes)
            data_all = load_data(idx_symbol, backtest_days)
            if data_all.empty:
                continue
            df_results_all = run_strategy(data_all)
            if df_results_all.empty:
                continue
            total_gain_all = df_results_all["Gain (pts)"].sum()
            df_results_all["Cumulative Gain"] = df_results_all["Gain (pts)"].cumsum()
            df_results_all["Index"] = idx_name
            all_comparison.append({
                "Index": idx_name,
                "Total Gain": total_gain_all,
                "Data": df_results_all
                })

        progress.empty()

        if not all_comparison:
            st.warning("No data found for any index to compare.")
        else:
            # Sort and select top 6 by total gain
            top6 = sorted(all_comparison, key=lambda x: x["Total Gain"], reverse=True)[:5]
        
            fig, ax = plt.subplots(figsize=(12, 6))
            for entry in top6:
                df_plot = entry["Data"]
                ax.plot(
                    df_plot["Date"], df_plot["Cumulative Gain"],
                    marker='o', linewidth=2, label=entry["Index"]
                )
        
            ax.set_title(f"Top 5 Indexes — Cumulative Gain (Last {backtest_days} Days)", fontsize=14, fontweight='bold')
            ax.set_xlabel("Date")
            ax.set_ylabel("Cumulative Gain (pts)")
            ax.legend(loc="upper left", fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.5)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)


# ---------------- SHOW LAST TRADE (ALL INDEXES) ----------------
if show_last_trade:
    st.info("Fetching last trade for all indexes... please wait ⏳")
    all_latest_trades = []

    for idx_name, idx_symbol in index_symbol_map.items():
        data = load_data(idx_symbol, 5)
        if data.empty:
            continue
        df_results = run_strategy(data)
        if df_results.empty:
            continue
        latest_trade = df_results.iloc[-1].copy()
        latest_trade["Index"] = idx_name
        all_latest_trades.append(latest_trade)

    if not all_latest_trades:
        st.error("No latest trades found for any index.")
    else:
        df_all_last = pd.DataFrame(all_latest_trades)
        df_all_last = df_all_last[
            ["Index", "Date", "Weekday", "Candle Color", "Direction", "Entry Time", "Exit Time", "Gain (pts)"]
        ]
        df_all_last.sort_values(by="Gain (pts)", ascending=False, inplace=True)

        # --- Matplotlib Table Visualization ---
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.axis('off')
        ax.set_title("📊 Latest Trade Summary for All Indexes", fontsize=16, fontweight='bold', pad=20)
        
        table_daily = ax.table(
            cellText=df_all_last.values,
            colLabels=df_all_last.columns,
            cellLoc='center',
            colWidths=[0.12]*len(df_all_last.columns),
            bbox=[0.01, 0.01, 0.98, 0.95]
        )
        table_daily.auto_set_font_size(False)
        table_daily.set_fontsize(10)

        col_candle = df_all_last.columns.get_loc("Candle Color")
        col_gain = df_all_last.columns.get_loc("Gain (pts)")

        for (row, col), cell in table_daily.get_celld().items():
            if row == 0:
                cell.set_facecolor('#4C72B0')
                cell.set_text_props(weight='bold', color='white')
            else:
                if row % 2 == 0:
                    cell.set_facecolor('#F5F5F5')
                else:
                    cell.set_facecolor('#FFFFFF')

                # Color Candle column
                if col == col_candle:
                    if cell.get_text().get_text() == "GREEN":
                        cell.get_text().set_color("green")
                        cell.get_text().set_weight("bold")
                    elif cell.get_text().get_text() == "RED":
                        cell.get_text().set_color("red")
                        cell.get_text().set_weight("bold")

                # Color Gain column
                if col == col_gain:
                    try:
                        val = float(cell.get_text().get_text())
                        if val > 0:
                            cell.set_facecolor('#C6EFCE')
                        elif val < 0:
                            cell.set_facecolor('#FFC7CE')
                        else:
                            cell.set_facecolor('#FFFFFF')
                    except:
                        pass

        plt.tight_layout()
        st.pyplot(fig)

# ---------------- RUN BACKTEST ----------------
elif run_backtest:
    st.info(f"Fetching {backtest_days} days of data for {selected_index} ({selected_symbol})...")
    data = load_data(selected_symbol, backtest_days)
    if data.empty:
        st.error("No data retrieved. Try another symbol or wait and retry.")
        st.stop()

    df_results = run_strategy(data)
    if df_results.empty:
        st.warning("No trades found.")
        st.stop()

    # -------- Summary --------
    df_results["Cumulative Gain"] = df_results["Gain (pts)"].cumsum()
    total_gain = df_results["Gain (pts)"].sum()
    avg_gain = df_results["Gain (pts)"].mean()
    win_days = (df_results["Gain (pts)"] > 0).sum()
    loss_days = (df_results["Gain (pts)"] < 0).sum()

    st.subheader(f"Summary for {selected_index}")
    st.write(f"- Total Days: {len(df_results)}")
    st.write(f"- Winning Days: {win_days}")
    st.write(f"- Losing Days: {loss_days}")
    st.write(f"- Avg Gain/Day: {avg_gain:.2f} pts")
    st.write(f"- Total Gain: {total_gain:.2f} pts")

    # -------- Weekday Summary --------
    weekday_summary = (
        df_results.groupby("Weekday")
        .agg(
            Winning_Days=("Gain (pts)", lambda x: (x > 0).sum()),
            Losing_Days=("Gain (pts)", lambda x: (x < 0).sum()),
            No_Trade_Days=("Direction", lambda x: (x == "No Trade").sum()),
            Total_Gain=("Gain (pts)", "sum"),
            Average_Gain=("Gain (pts)", "mean"),
        )
        .reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
        .fillna(0)
    )

    st.subheader("📅 Weekday Performance")
    st.dataframe(weekday_summary.style.format("{:.2f}"))

    # -------- ALL VISUALIZATIONS (from old code) --------
    symbol = selected_symbol
    def plot_win_loss_no_trade(weekday_summary):
        fig, ax = plt.subplots(figsize=(10,6))
        weekday_summary[["Winning_Days", "Losing_Days", "No_Trade_Days"]].plot(
            kind="bar", stacked=True, color=["#5CFF5C", "#FF2E2E", "gray"], ax=ax)
        ax.set_title(f"{symbol} - Weekday-wise Win/Loss/No-Trade Days")
        ax.set_ylabel("Number of Days")
        ax.set_xlabel("Weekday")
        plt.xticks(rotation=0)
        st.pyplot(fig)

    def plot_wins_losses_weekday(weekday_summary):
        fig, ax = plt.subplots(figsize=(10,6))
        idx = weekday_summary.index
        width = 0.3
        x = np.arange(len(idx))
        ax.bar(x - width, weekday_summary['Winning_Days'], width=width, label='Winning Days', color='green')
        ax.bar(x, weekday_summary['Losing_Days'], width=width, label='Losing Days', color='red')
        ax.bar(x + width, weekday_summary['No_Trade_Days'], width=width, label='No Trade', color='gray')
        ax.set_xticks(x)
        ax.set_xticklabels(idx)
        ax.set_ylabel('Number of Days')
        ax.set_title('Wins/Losses/No Trade by Weekday')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

    def plot_total_gain_weekday(weekday_summary):
        fig, ax = plt.subplots(figsize=(8,5))
        weekday_summary["Total_Gain"].plot(kind="bar", color="skyblue", ax=ax)
        ax.set_title(f"{symbol} - Total Gain per Weekday")
        ax.set_ylabel("Total Gain (pts)")
        ax.set_xlabel("Weekday")
        plt.xticks(rotation=0)
        st.pyplot(fig)

    def plot_avg_gain_weekday(weekday_summary):
        fig, ax = plt.subplots(figsize=(8,5))
        weekday_summary["Average_Gain"].plot(kind="line", marker="o", color="blue", ax=ax)
        ax.set_title(f"{symbol} - Average Gain per Weekday")
        ax.set_ylabel("Average Gain (pts)")
        ax.set_xlabel("Weekday")
        ax.grid(True)
        st.pyplot(fig)

    def plot_cumulative_gain(df_results):
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(df_results['Date'], df_results['Cumulative Gain'], marker='o')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Gain (pts)')
        ax.set_title('Cumulative Gain Over Time')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

    def plot_trade_outcome_pie(df_results):
        outcomes = [
            (df_results["Gain (pts)"] > 0).sum(),
            (df_results["Gain (pts)"] < 0).sum(),
            (df_results["Direction"] == "No Trade").sum()
        ]
        labels = ['Winning Days', 'Losing Days', 'No Trade']
        fig, ax = plt.subplots(figsize=(6,6))
        ax.pie(outcomes, labels=labels, autopct='%1.1f%%', startangle=140, wedgeprops=dict(width=0.5))
        ax.set_title('Trade Outcomes Distribution')
        st.pyplot(fig)

    def plot_weekday_percentage_stacked_bar(weekday_summary):
        weekday_percent = weekday_summary[["Winning_Days", "Losing_Days", "No_Trade_Days"]].div(
            weekday_summary[["Winning_Days", "Losing_Days", "No_Trade_Days"]].sum(axis=1), axis=0
        ) * 100
        fig, ax = plt.subplots(figsize=(10,6))
        weekday_percent.plot(kind="bar", stacked=True, color=["green", "red", "gray"], ax=ax)
        ax.set_ylabel("Percentage (%)")
        ax.set_xlabel("Weekday")
        ax.set_title(f"{symbol} - Weekday-wise Win/Loss/No-Trade Percentage")
        plt.xticks(rotation=0)
        ax.legend(loc="upper right")
        plt.tight_layout()
        st.pyplot(fig)

    def plot_calendar_heatmap(df_results):
        df_results_copy = df_results.copy()
        df_results_copy['Date'] = pd.to_datetime(df_results_copy['Date'])
        df_results_copy.set_index('Date', inplace=True)
        df_results_copy['Year'] = df_results_copy.index.year
        df_results_copy['Month'] = df_results_copy.index.month
        df_results_copy['Day'] = df_results_copy.index.day
        df_results_copy['Weekday'] = df_results_copy.index.weekday
        latest_year = df_results_copy['Year'].max()
        latest_month = df_results_copy[df_results_copy['Year'] == latest_year]['Month'].max()
        month_data = df_results_copy[(df_results_copy['Year'] == latest_year) & (df_results_copy['Month'] == latest_month)]
        if not month_data.empty:
            pivot = month_data.pivot(index='Day', columns='Weekday', values='Gain (pts)')
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(pivot, annot=True, fmt=".2f", cmap='RdYlGn', center=0,
                        linewidths=0.5, cbar_kws={'label': 'Gain (pts)'}, ax=ax)
            ax.set_title(f'Calendar Heatmap of Daily Gains: {latest_year}-{latest_month:02d}')
            ax.set_xlabel('Weekday (Mon=0)')
            ax.set_ylabel('Day of Month')
            st.pyplot(fig)

    def plot_winning_percentage_donut(weekday_summary):
        ws = weekday_summary.copy()
        ws['Winning_Percentage'] = (
            ws['Winning_Days'] /
            (ws['Winning_Days'] + ws['Losing_Days']) * 100
        ).round(2)
        sizes = ws['Winning_Percentage']
        labels = ws.index
        colors = ['#A8E6CF', '#DCEDC1', '#FFD3B6', '#FFAAA5', '#FF8B94']
        fig, ax = plt.subplots(figsize=(8,8))
        wedges, _, _ = ax.pie(
            sizes, labels=None, autopct='%1.1f%%', startangle=90, colors=colors,
            wedgeprops=dict(width=0.4, edgecolor='white'), pctdistance=0.75,
            textprops=dict(weight='bold', color='#333333', fontsize=12)
        )
        for i, p in enumerate(wedges):
            angle = (p.theta2 - p.theta1) / 2 + p.theta1
            y = 1.2 * np.sin(np.deg2rad(angle))
            x = 1.2 * np.cos(np.deg2rad(angle))
            ax.text(x, y, f"{labels[i]}", ha='center', va='center', fontsize=12, weight='bold', color='#555555')
        ax.set_title(f"{symbol} - Winning Percentage per Weekday", fontsize=16, fontweight='bold')
        ax.axis('equal')
        st.pyplot(fig)

    def plot_losing_percentage_donut(weekday_summary):
        ws = weekday_summary.copy()
        ws['Losing_Percentage'] = (
            ws['Losing_Days'] /
            (ws['Winning_Days'] + ws['Losing_Days']) * 100
        ).round(2)
        sizes = ws['Losing_Percentage']
        labels = ws.index
        colors = ['#FFCDD2', '#EF9A9A', '#E57373', '#EF5350', '#F44336']
        fig, ax = plt.subplots(figsize=(8,8))
        wedges, _, _ = ax.pie(
            sizes, labels=None, autopct='%1.1f%%', startangle=90, colors=colors,
            wedgeprops=dict(width=0.4, edgecolor='white'), pctdistance=0.75,
            textprops=dict(weight='bold', color='#333333', fontsize=12)
        )
        for i, p in enumerate(wedges):
            angle = (p.theta2 - p.theta1) / 2 + p.theta1
            y = 1.2 * np.sin(np.deg2rad(angle))
            x = 1.2 * np.cos(np.deg2rad(angle))
            ax.text(x, y, f"{labels[i]}", ha='center', va='center', fontsize=12, weight='bold', color='#555555')
        ax.set_title(f"{symbol} - Losing Percentage per Weekday", fontsize=16, fontweight='bold')
        ax.axis('equal')
        st.pyplot(fig)

    def plot_winning_losing_percentage_stackbar(weekday_summary):
        ws = weekday_summary.copy()
        ws['Total_Trades'] = ws['Winning_Days'] + ws['Losing_Days']
        ws['Winning_Percentage'] = (ws['Winning_Days'] / ws['Total_Trades'] * 100).round(2)
        ws['Losing_Percentage'] = (ws['Losing_Days'] / ws['Total_Trades'] * 100).round(2)
        fig, ax = plt.subplots(figsize=(10,6))
        ax.bar(ws.index, ws['Winning_Percentage'], label='Winning %', color='#79f299')
        ax.bar(ws.index, ws['Losing_Percentage'], bottom=ws['Winning_Percentage'], label='Losing %', color='#f75f54')
        for i, row in ws.iterrows():
            ax.text(i, row['Winning_Percentage']/2, f"{row['Winning_Percentage']}%", ha='center', va='center', color='white', fontweight='bold')
            ax.text(i, row['Winning_Percentage'] + row['Losing_Percentage']/2, f"{row['Losing_Percentage']}%", ha='center', va='center', color='white', fontweight='bold')
        ax.set_ylabel('Percentage (%)')
        ax.set_title(f"{symbol} - Winning vs Losing Percentage per Weekday", fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        plt.xticks(rotation=0)
        plt.ylim(0, 100)
        plt.tight_layout()
        st.pyplot(fig)

    def plot_final_summary_tables():
        summary_data = {
            "Metric": ["Total Days Tested", "Winning Days", "Losing Days", "Average Gain/Day", "Total Net Gain"],
            "Value": [len(df_results), win_days, loss_days, f"{avg_gain:.3f} pts", f"{total_gain:.3f} pts"]
        }
        summary_df = pd.DataFrame(summary_data)
        weekday_reset = weekday_summary.reset_index().copy()
        weekday_reset = weekday_reset.round(3)
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        ax.set_title(f"Final Backtest Summary — {symbol}", fontsize=16, fontweight='bold', pad=20)
        col_width_summary = [0.5, 0.3]
        col_width_weekday = [0.18] * len(weekday_reset.columns)
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
        for (row, col), cell in table_summary.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#4C72B0')
            else:
                cell.set_facecolor('#EAF2FA')
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
        for (row, col), cell in table_weekday.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#2C7BB6')
            elif row % 2 == 0:
                cell.set_facecolor('#F5F5F5')
            else:
                cell.set_facecolor('#FFFFFF')
        plt.tight_layout()
        st.pyplot(fig)

    def plot_latest_30days_table():
        df_display = df_results.tail(30).copy()
        df_display['Entry Time'] = df_display['Entry Time'].astype(str)
        df_display['Exit Time'] = df_display['Exit Time'].astype(str)
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
        col_candle = df_display.columns.get_loc("Candle Color")
        col_gain = df_display.columns.get_loc("Gain (pts)")
        for (row, col), cell in table_daily.get_celld().items():
            if row == 0:
                cell.set_facecolor('#4C72B0')
                cell.set_text_props(weight='bold', color='white')
            else:
                if row % 2 == 0:
                    cell.set_facecolor('#F5F5F5')
                else:
                    cell.set_facecolor('#FFFFFF')
                if col == col_candle:
                    if cell.get_text().get_text() == "GREEN":
                        cell.get_text().set_color("green")
                        cell.get_text().set_weight("bold")
                    elif cell.get_text().get_text() == "RED":
                        cell.get_text().set_color("red")
                        cell.get_text().set_weight("bold")
                if col == col_gain:
                    try:
                        val = float(cell.get_text().get_text())
                        if val > 0:
                            cell.set_facecolor('#C6EFCE')
                        elif val < 0:
                            cell.set_facecolor('#FFC7CE')
                        else:
                            cell.set_facecolor('#FFFFFF')
                    except:
                        pass
        plt.tight_layout()
        st.pyplot(fig)

    def plot_top_worst_weekdays():
        top_weekdays = weekday_summary.sort_values("Average_Gain", ascending=False).head(3)
        worst_weekdays = weekday_summary.sort_values("Average_Gain").head(3)
        top_weekdays_plot = top_weekdays[['Average_Gain', 'Total_Gain']].copy()
        top_weekdays_plot['Average_Gain'] = top_weekdays_plot['Average_Gain'].round(3)
        worst_weekdays_plot = worst_weekdays[['Average_Gain', 'Total_Gain']].copy()
        worst_weekdays_plot['Average_Gain'] = worst_weekdays_plot['Average_Gain'].round(3)
        mpl.rcParams['font.family'] = 'Segoe UI Emoji'
        fig, axes = plt.subplots(2, 1, figsize=(8, 6))
        plt.subplots_adjust(hspace=0.6)
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
        st.pyplot(fig)

    def plot_strategy_vs_nsei(df_results):
        df_resultsc = df_results.copy()
        df_resultsc['Date'] = pd.to_datetime(df_resultsc['Date'])
        df_resultsc.set_index('Date', inplace=True)
        if 'Cumulative Gain' not in df_resultsc.columns:
            df_resultsc['Cumulative Gain'] = df_resultsc['Gain (pts)'].cumsum()
        nsei = yf.download("^NSEI", interval="5m", period="60d", progress=False, auto_adjust=True)
        nsei['Close'] = nsei['Close'].ffill()
        nsei_daily = nsei['Close'].resample('D').last()
        nsei_cum = nsei_daily.pct_change(fill_method=None).cumsum() * 1000
        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(df_resultsc.index, df_resultsc['Cumulative Gain'], marker='o', label='Strategy', color='green')
        ax.plot(nsei_cum.index, nsei_cum, marker='o', label='NSEI % Change x1000', color='royalblue')
        ax.set_title(f"{symbol} Strategy vs NSEI Index", fontsize=14, fontweight='bold')
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Gain / Relative Performance")
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

    # Call all visualization functions
    st.header("Visualizations")
    plot_final_summary_tables()
    plot_latest_30days_table()
    plot_top_worst_weekdays()
    plot_win_loss_no_trade(weekday_summary)
    plot_wins_losses_weekday(weekday_summary)
    plot_total_gain_weekday(weekday_summary)
    plot_avg_gain_weekday(weekday_summary)
    plot_cumulative_gain(df_results)
    plot_trade_outcome_pie(df_results)
    plot_weekday_percentage_stacked_bar(weekday_summary)
    plot_calendar_heatmap(df_results)
    plot_winning_percentage_donut(weekday_summary)
    plot_losing_percentage_donut(weekday_summary)
    plot_winning_losing_percentage_stackbar(weekday_summary)
    plot_strategy_vs_nsei(df_results)

else:
    st.write("Press 'Run Backtest' or 'Show Last Trade (All Indexes)' to continue.")
