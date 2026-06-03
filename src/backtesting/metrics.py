"""Summary metrics for strategy backtests."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return np.nan
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def _cagr(equity: pd.Series, dates: pd.Series | pd.Index) -> float:
    if len(equity) < 2:
        return np.nan
    start_equity = float(equity.iloc[0])
    end_equity = float(equity.iloc[-1])
    if start_equity <= 0:
        return np.nan
    dt_index = pd.to_datetime(pd.Index(dates))
    years = (dt_index[-1] - dt_index[0]).days / 365.25
    if years <= 0:
        return np.nan
    return float((end_equity / start_equity) ** (1.0 / years) - 1.0)


def _sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 2:
        return np.nan
    stdev = clean.std(ddof=1)
    if stdev == 0 or pd.isna(stdev):
        return np.nan
    return float(clean.mean() / stdev * np.sqrt(periods_per_year))


def summarize_backtest(
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    initial_capital: float,
    periods_per_year: int = 252,
) -> dict:
    """Compute a compact summary for one backtest run."""
    if equity_curve.empty:
        return {
            "initial_capital": initial_capital,
            "final_equity": initial_capital,
            "total_return": 0.0,
            "cagr": np.nan,
            "max_drawdown": np.nan,
            "sharpe": np.nan,
            "return_over_drawdown": np.nan,
            "profit_factor": np.nan,
            "win_rate": np.nan,
            "average_win": np.nan,
            "average_loss": np.nan,
            "expectancy_pct": np.nan,
            "expectancy_dollars": np.nan,
            "number_of_trades": 0,
            "exposure_time": 0.0,
            "average_bars_held": np.nan,
            "best_trade": np.nan,
            "worst_trade": np.nan,
        }

    equity = equity_curve["equity"].astype(float)
    returns = equity.pct_change()
    final_equity = float(equity.iloc[-1])
    total_return = final_equity / initial_capital - 1.0
    max_dd = _max_drawdown(equity)
    return_over_drawdown = (
        float(total_return / abs(max_dd))
        if pd.notna(max_dd) and max_dd != 0
        else np.nan
    )

    if trades.empty:
        gross_profit = 0.0
        gross_loss = 0.0
        winning = pd.DataFrame()
        losing = pd.DataFrame()
        trade_returns = pd.Series(dtype=float)
        trade_pnl = pd.Series(dtype=float)
    else:
        trade_returns = trades["net_return"].astype(float)
        trade_pnl = trades["net_pnl"].astype(float)
        winning = trades[trades["net_pnl"] > 0]
        losing = trades[trades["net_pnl"] < 0]
        gross_profit = float(winning["net_pnl"].sum()) if not winning.empty else 0.0
        gross_loss = float(losing["net_pnl"].sum()) if not losing.empty else 0.0

    if gross_loss < 0:
        profit_factor = gross_profit / abs(gross_loss)
    elif gross_profit > 0:
        profit_factor = np.inf
    else:
        profit_factor = np.nan

    number_of_trades = int(len(trades))
    exposure_time = (
        float((equity_curve["position_side"].astype(float) != 0).mean())
        if "position_side" in equity_curve.columns
        else np.nan
    )

    return {
        "initial_capital": float(initial_capital),
        "final_equity": final_equity,
        "total_return": float(total_return),
        "cagr": _cagr(equity, equity_curve["date"]),
        "max_drawdown": max_dd,
        "sharpe": _sharpe(returns, periods_per_year),
        "return_over_drawdown": return_over_drawdown,
        "profit_factor": float(profit_factor) if pd.notna(profit_factor) else np.nan,
        "win_rate": float((trade_returns > 0).mean()) if number_of_trades else np.nan,
        "average_win": float(winning["net_return"].mean()) if not winning.empty else np.nan,
        "average_loss": float(losing["net_return"].mean()) if not losing.empty else np.nan,
        "expectancy_pct": float(trade_returns.mean()) if number_of_trades else np.nan,
        "expectancy_dollars": float(trade_pnl.mean()) if number_of_trades else np.nan,
        "number_of_trades": number_of_trades,
        "exposure_time": exposure_time,
        "average_bars_held": (
            float(trades["bars_held"].mean()) if number_of_trades else np.nan
        ),
        "best_trade": float(trade_returns.max()) if number_of_trades else np.nan,
        "worst_trade": float(trade_returns.min()) if number_of_trades else np.nan,
    }
