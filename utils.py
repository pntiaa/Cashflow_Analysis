import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List

# --- PriceDeck Setup ---

def rounding_decorator(func):
    def wrapper(self, *args, **kwargs) -> Dict[int, float]:
        price_by_year = func(self, *args, **kwargs)
        if not isinstance(price_by_year, dict):
            raise TypeError("The decorated function must return Dict[int, float].")
        return {year: round(price, 2) for year, price in price_by_year.items()}
    return wrapper

class PriceDeck:
    def __init__(self,
         start_year: int = 2025, end_year: int = 2075,
         oil_price_initial: float = 70.0, gas_price_initial: float = 8.0, inflation_rate: float = 0.015,
         flat_after_year: int = None,
         oil_price_by_year: Dict[int, float] = None,
         gas_price_by_year: Dict[int, float] = None):

        self.start_year = start_year
        self.end_year = end_year
        self.inflation_rate = inflation_rate
        self.flat_after_year = flat_after_year
        self.years: List[int] = range(start_year, end_year+1, 1)
        self.oil_price_by_year: Dict[int, float] = {}
        self.gas_price_by_year: Dict[int, float] = {}

        self.oil_price_by_year = self._setting_initial_oil_price(oil_price_initial)
        self.gas_price_by_year = self._setting_initial_gas_price(gas_price_initial)

        if oil_price_by_year:
            for y, price in oil_price_by_year.items():
                self.oil_price_by_year[y] = price

        if gas_price_by_year:
            for y, price in gas_price_by_year.items():
                self.gas_price_by_year[y] = price

        if flat_after_year:
            self._setting_flat_price(flat_after_year)

    @rounding_decorator
    def _setting_initial_oil_price(self, oil_price_initial) -> Dict[int, float]:
        calculated_prices = {}
        for y in self.years:
            years_from_base = (y - self.start_year)
            calculated_prices[y] = oil_price_initial * ((1 + self.inflation_rate) ** years_from_base)
        return calculated_prices

    @rounding_decorator
    def _setting_initial_gas_price(self, gas_price_initial) -> Dict[int, float]:
        calculated_prices = {}
        for y in self.years:
            years_from_base = (y - self.start_year)
            calculated_prices[y] = gas_price_initial * ((1 + self.inflation_rate) ** years_from_base)
        return calculated_prices

    def _setting_flat_price(self, flat_after_year: int=None):
        for y in self.years:
            if y > flat_after_year:
                self.oil_price_by_year[y] = self.oil_price_by_year.get(flat_after_year, self.oil_price_by_year[max(self.oil_price_by_year.keys())])
                self.gas_price_by_year[y] = self.gas_price_by_year.get(flat_after_year, self.gas_price_by_year[max(self.gas_price_by_year.keys())])
        return self.oil_price_by_year, self.gas_price_by_year

# --- Plotting Functions ---

def plot_cash_flow_profile_plotly(cf, width=1200, height=800):
    years = cf.all_years
    rev = np.array([cf.annual_revenue.get(y, 0.0) for y in years])
    royalty = np.array([cf.annual_royalty.get(y, 0.0) for y in years])
    cap = np.array([cf.annual_capex.get(y, 0.0) for y in years])
    opx = np.array([cf.annual_opex.get(y, 0.0) for y in years])
    abx = np.array([cf.annual_abex.get(y, 0.0) for y in years])
    tax = np.array([cf.annual_total_tax.get(y, 0.0) for y in years])
    net = np.array([cf.annual_net_cash_flow.get(y, 0.0) for y in years])
    cum = np.array([cf.cumulative_cash_flow.get(y, 0.0) for y in years])
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Annual Cash Flow Components",
            "Cumulative Cash Flow (After Tax)",
            "Production Profile",
            "Commodity Prices"
        ),
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy", "secondary_y": True}]
        ],
        column_widths=[0.6, 0.4],
        row_heights =[0.6, 0.4],
        horizontal_spacing=0.1,
        vertical_spacing=0.2,
    )

    # 1. Annual
    fig.add_trace( go.Bar( x=years, y=rev, name='Revenue', marker_color='rgb(102,194,165)'), row=1, col=1)
    fig.add_trace( go.Bar( x=years, y=-royalty, name='Royalty', marker_color='rgb(252,141,98)' ), row=1, col=1)
    fig.add_trace( go.Bar( x=years, y=-(cap + opx + abx), name='Costs', marker_color='rgb(141,160,203)' ), row=1, col=1)
    fig.add_trace( go.Bar( x=years, y=-tax, name='Taxes', marker_color='rgb(231,138,195)' ), row=1, col=1)
    fig.add_trace(go.Scatter(x=years, y=net, name='Net Flow', mode='lines+markers', line=dict(color='black', width=2)), row=1, col=1)
    fig.update_layout(barmode='relative')

    # 2. Cumulative
    fig.add_trace(go.Scatter(
        x=years, y=cum, name='Cumulative', mode='lines',
        line=dict(color='purple', width=3), fill='tozeroy'
    ), row=1, col=2)
    fig.add_hline(y=0.0, line_dash="dash", line_color="red", row=1, col=2)

    # 3. Production
    oil_prod = [cf.oil_production_by_year.get(y, 0.0) for y in years]
    gas_prod = [cf.gas_production_by_year.get(y, 0.0) for y in years]
    fig.add_trace(go.Bar(x=years, y=gas_prod, name='Gas (BCF)', marker_color='lightblue'), row=2, col=1)
    
    # 4. Prices
    oilp = [cf.oil_price_by_year.get(y, 0.0) for y in years]
    gasp = [cf.gas_price_by_year.get(y, 0.0) for y in years]
    fig.add_trace(go.Scatter(x=years, y=oilp, name='Oil ($/bbl)', line=dict(color='green')), row=2, col=2, secondary_y=False)
    fig.add_trace(go.Scatter(x=years, y=gasp, name='Gas ($/mcf)', line=dict(color='red')), row=2, col=2, secondary_y=True)

    fig.update_layout(height=height, width=width, title_text="Economic Results", template='plotly_white')
    return fig

def summary_plot(cf, width=1200, height=700):
    years = cf.all_years
    rev = np.array([cf.annual_revenue.get(y, 0.0) for y in years])
    cap = np.array([cf.annual_capex.get(y, 0.0) for y in years])
    opx = np.array([cf.annual_opex.get(y, 0.0) for y in years])
    abx = np.array([cf.annual_abex.get(y, 0.0) for y in years])
    tax = np.array([cf.annual_total_tax.get(y, 0.0) for y in years])
    net = np.array([cf.annual_net_cash_flow.get(y, 0.0) for y in years])
    cum = np.array([cf.cumulative_cash_flow.get(y, 0.0) for y in years])

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Key Metrics", "Price Trends", "Annual & Cumulative Cash Flow"),
        specs=[
            [{"type": "table"}, {"type": "xy","secondary_y": True}],
            [{"type": "xy", "colspan": 2, "secondary_y": True}, None],
        ],
        column_widths=[0.4, 0.6],
        row_heights =[0.4, 0.6],
        horizontal_spacing=0.1,
        vertical_spacing=0.2,
    )

    # Table
    metrics = ['Net Cash Flow', 'NPV', 'IRR']
    values = [
        f"{cum[-1]:,.1f} MM$",
        f"{cf.npv:,.1f} MM$",
        f"{cf.irr*100:.1f}%" if cf.irr else "N/A"
    ]
    fig.add_trace(go.Table(
        header=dict(values=['Metric', 'Value'], fill_color='paleturquoise', align='left'),
        cells=dict(values=[metrics, values], fill_color='lavender', align='left')
    ), row=1, col=1)

    # Prices
    oilp = [cf.oil_price_by_year.get(y, 0.0) for y in years]
    gasp = [cf.gas_price_by_year.get(y, 0.0) for y in years]
    fig.add_trace(go.Scatter(x=years, y=oilp, name='Oil Price', line=dict(color='green')), row=1, col=2, secondary_y=False)
    fig.add_trace(go.Scatter(x=years, y=gasp, name='Gas Price', line=dict(color='red')), row=1, col=2, secondary_y=True)

    # Main Chart
    color_palette = px.colors.qualitative.Set3
    fig.add_trace(go.Bar(x=years, y=rev, name='Revenue', marker_color=color_palette[0]), row=2, col=1)
    fig.add_trace(go.Bar(x=years, y=-cap, name='CAPEX', marker_color=color_palette[1]), row=2, col=1)
    fig.add_trace(go.Bar(x=years, y=-opx, name='OPEX', marker_color=color_palette[2]), row=2, col=1)
    fig.add_trace(go.Bar(x=years, y=-tax, name='Tax', marker_color=color_palette[3]), row=2, col=1)
    fig.add_trace(go.Scatter(x=years, y=net, name='NCF', line=dict(color='black', width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=years, y=cum, name='Cumulative', line=dict(color='purple', dash='dot')), row=2, col=1, secondary_y=True)

    fig.update_layout(height=height, width=width, template='plotly_white', barmode='relative')
    return fig
