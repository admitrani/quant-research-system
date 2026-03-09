import backtrader as bt
import logging

logger = logging.getLogger(__name__)


class MLStrategyV1(bt.Strategy):

    params = (
        ("entry_threshold", None),
        ("exit_threshold", None),
        ("risk_fraction", None),
        ("commission", None),
        ("slippage", None),
        ("buffer", None),
        ("max_positions", None),
        ("minimum_capital", None),
        ("max_drawdown_limit", None),
        ("min_position_size", None),
    )

    def __init__(self):

        self.prob = self.datas[0].ml_prob
        self.order = None

        self.bars_in_market = 0
        self.total_bars = 0
        self.trade_durations = []

        self.stop_trading_triggered = False
        self.peak_equity = self.broker.getvalue()

        logger.info(
            f"MLStrategyV1 initialised | "
            f"entry_threshold: {self.p.entry_threshold} | "
            f"exit_threshold: {self.p.exit_threshold} | "
            f"risk_fraction: {self.p.risk_fraction} | "
            f"minimum_capital: {self.p.minimum_capital} | "
            f"max_drawdown_limit: {self.p.max_drawdown_limit} | "
            f"min_position_size: {self.p.min_position_size}"
        )

    def notify_order(self, order):

        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Canceled, order.Margin, order.Rejected]:
            logger.warning(f"ORDER FAILED | Status: {order.getstatusname()} | Size: {order.size}")

        self.order = None

    def next(self):

        # Prevent multiple orders
        if self.order:
            return

        # Track bars
        self.total_bars += 1
        if self.position.size > 0:
            self.bars_in_market += 1

        # Current equity
        equity = self.broker.getvalue()

        # Track equity peak
        if equity > self.peak_equity:
            self.peak_equity = equity

        drawdown = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0

        # Stop trading conditions
        if equity < self.p.minimum_capital:
            return

        if drawdown > self.p.max_drawdown_limit:
            if not self.stop_trading_triggered:
                dt = self.datas[0].datetime.datetime(0)
                logger.warning(
                    f"STOP TRADING | Drawdown {drawdown:.2%} exceeded limit "
                    f"{self.p.max_drawdown_limit:.2%} at {dt}"
                )
                self.stop_trading_triggered = True
            return

        # ML probability signal
        prob = self.prob[0]
        signal = prob > self.p.entry_threshold
        current_position = self.position.size > self.p.min_position_size
        exit_signal = prob < self.p.exit_threshold

        # Entry logic
        if signal and not current_position:
            cash = self.broker.getcash()
            price = self.data.close[0]
            total_cost_factor = 1 + self.p.commission + self.p.slippage + self.p.buffer
            size = (cash * self.p.risk_fraction) / (price * total_cost_factor)
            if size > self.p.min_position_size:
                self.order = self.buy(size=size)

        # Exit logic
        elif exit_signal and current_position:
            self.order = self.close()

    def notify_trade(self, trade):

        if trade.isclosed:
            duration = trade.barclose - trade.baropen
            self.trade_durations.append(duration)