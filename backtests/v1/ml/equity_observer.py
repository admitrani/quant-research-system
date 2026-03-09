import backtrader as bt


class EquityObserver(bt.Observer):

    lines = ("equity",)

    plotinfo = dict(plot=True, subplot=True)

    def next(self):
        self.lines.equity[0] = self._owner.broker.getvalue()
