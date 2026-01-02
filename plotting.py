import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import matplotlib.pyplot as plt
from typing import Dict, List
from matplotlib import colors as mcolors # HEX 코드를 RGB로 변환하기 위해 사용


def plot_production_profile(cf, show: bool = True):
    """
    Plot production profile using Plotly.
    Relies on cf.production_profile being present.
    """
    if not cf.annual_gas_production:
        raise ValueError("Production profile not calculated. Run calculate_production_profile() first.")

    years = sorted(cf.annual_gas_production.keys())
    gas_production_vals = [cf.annual_gas_production.get(y, 0.0) for y in years]
    oil_production_vals = [cf.annual_oil_production.get(y, 0.0) for y in years]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=gas_production_vals, mode='lines', name='Gas Production'))
    fig.add_trace(go.Scatter(x=years, y=oil_production_vals, mode='lines', name='Oil Production'))
    fig.update_layout(title='Production Profile', xaxis_title='Year', yaxis_title='Production (MM barrels)')
    return fig


def plot_dev_cost_profile(dev):
    years = sorted(dev.cost_years)
    capex_vals = [dev.annual_capex.get(y, 0.0) for y in years]
    opex_vals = [dev.annual_opex.get(y, 0.0) for y in years]
    abex_vals = [dev.annual_abex.get(y, 0.0) for y in years]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=years, y=capex_vals, name='CAPEX'))
    fig.add_trace(go.Bar(x=years, y=opex_vals, name='OPEX'))
    fig.add_trace(go.Bar(x=years, y=abex_vals, name='ABEX'))
    
    fig.update_layout(
        barmode='stack',
        title=f"Annual Expenditure Forecast (MM$)",
        xaxis_title="Year",
        yaxis_title="MM$",
        legend_title="Cost Category"
    )
    return fig
    
def plot_dev_prod_profile(dev):    
    p_years = sorted(dev.production_years)
    gas_vals = [dev.annual_gas_production.get(y, 0.0) for y in p_years]
    oil_vals = [dev.annual_oil_production.get(y, 0.0) for y in p_years]
    
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=p_years, y=gas_vals, mode='lines+markers', name='Gas (BCF)'))
    fig_p.add_trace(go.Scatter(x=p_years, y=oil_vals, mode='lines+markers', name='Oil/Cond. (MMbbl)'))
    
    fig_p.update_layout(
        title="Annual Production Profile",
        xaxis_title="Year",
        yaxis_title="Volume",
        legend_title="Product"
    )
    return fig_p

def plot_price(cf):
    """
    Plot price profile using Plotly.
    Relies on cf.price_profile being present.
    """
    if not cf.gas_price_by_year:
        raise ValueError("Price profile not calculated. Run calculate_price_profile() first.")

    years = sorted(cf.gas_price_by_year.keys())
    gas_price_vals = [cf.gas_price_by_year.get(y, 0.0) for y in years]
    oil_price_vals = [cf.oil_price_by_year.get(y, 0.0) for y in years]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=gas_price_vals, mode='lines', name='Gas Price'))
    fig.add_trace(go.Scatter(x=years, y=oil_price_vals, mode='lines', name='Oil Price'))
    fig.update_layout(title='Price Profile', xaxis_title='Year', yaxis_title='Price (USD/MM barrel)')
    return fig


# DevelopmentCost class plotting functions
def plot_cost_profile(dev_cost):
    """
    Plot annual capex/opex/abex stacked bar and cumulative curve using Plotly.
    Relies on dev_cost.total_annual_costs, dev_cost.annual_capex, dev_cost.annual_opex, dev_cost.annual_abex being present.
    """
    if not dev_cost.total_annual_costs:
        raise ValueError("Costs not calculated. Run calculate_total_costs() first.")

    years = sorted(dev_cost.total_annual_costs.keys())
    capex_vals = [dev_cost.annual_capex.get(y, 0.0) for y in years]
    opex_vals = [dev_cost.annual_opex.get(y, 0.0) for y in years]
    abex_vals = [dev_cost.annual_abex.get(y, 0.0) for y in years]
    total_vals = [dev_cost.total_annual_costs.get(y, 0.0) for y in years]
    cum_vals = [dev_cost.cumulative_costs.get(y, 0.0) for y in years]

    # Plot 1: stacked bars
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

    ax1.bar(years, capex_vals, label='CAPEX')
    ax1.bar(years, opex_vals, bottom=capex_vals, label='OPEX')
    bottom_for_abex = [c + o for c, o in zip(capex_vals, opex_vals)]
    ax1.bar(years, abex_vals, bottom=bottom_for_abex, label='ABEX')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Cost (MM$)')
    ax1.set_title(f'Annual Cost Profile - {dev_cost.development_case}')
    y_max = max(bottom_for_abex)*1.2
    y_max_digit = int(np.log10(y_max))
    y_max_grid = np.round(y_max, y_max_digit*-1)
    ax1.set_yticks(np.linspace(0,y_max_grid, 6))
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 3: drilled wells
    ax3 = ax1.twinx()
    # Align drilled_wells with the full timeline (years)
    drilled_wells_aligned = [dev_cost.yearly_drilling_schedule.get(y, 0) for y in years]
    ax3.plot(years, drilled_wells_aligned, marker='o', color='blue', label='Drilled Wells')
    ax3.set_ylabel('Drilled Wells', color='blue')
    ax3.tick_params(axis='y', labelcolor='blue')
    ax3.legend(loc='upper right')
    y_max = max(drilled_wells_aligned)*2
    ax3.set_yticks(np.arange(0, y_max, 2))

    # Plot 2: cumulative
    ax2.plot(years, cum_vals, marker='o', linestyle='-')
    ax2.fill_between(years, cum_vals, alpha=0.2)
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Cumulative Cost (MM$)')
    ax2.set_title('Cumulative Cost')
    ax2.grid(True, alpha=0.3)
    ax2.annotate(f'Total: {cum_vals[-1]:.2f} MM', xy=(years[-1], cum_vals[-1]), xytext=(10, 10),
                    textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.tight_layout()
    return fig


def plot_total_annual_costs(dev_cost):
    """
    Plots the total annual costs as a bar chart.
    """
    if not dev_cost.total_annual_costs:
        raise ValueError("Total annual costs have not been calculated. Call calculate_total_costs() first.")

    years = list(dev_cost.total_annual_costs.keys())
    costs = list(dev_cost.total_annual_costs.values())

    plt.figure(figsize=(6, 4))
    plt.bar(years, costs, color='skyblue')
    plt.xlabel('Year')
    plt.ylabel('Total Annual Costs (MM$)')
    plt.title('Total Annual Project Costs')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def get_rgba_from_hex(hex_or_rgb_code, alpha=0.5):
    """
    HEX 또는 RGB 색상 코드를 받아 지정된 투명도(alpha)가 적용된 RGBA 문자열을 반환합니다.
    """
    if hex_or_rgb_code.startswith('rgb('):
        # Parse rgb(R,G,B) format
        parts = hex_or_rgb_code[4:-1].split(',')
        r = int(parts[0].strip())
        g = int(parts[1].strip())
        b = int(parts[2].strip())
    else:
        # Assume HEX format if not rgb
        rgb_float = mcolors.hex2color(hex_or_rgb_code)
        r = int(rgb_float[0] * 255)
        g = int(rgb_float[1] * 255)
        b = int(rgb_float[2] * 255)
    return f'rgba({r}, {g}, {b}, {alpha})'

    
def plot_cash_flow_profile_plotly(cf, width=1200, height=800):
    years = cf.all_years
    rev = np.array([cf.annual_revenue.get(y, 0.0) for y in years])
    royalty = np.array([cf.annual_royalty.get(y, 0.0) for y in years])
    cap = np.array([cf.annual_capex_inflated.get(y, 0.0) for y in years])
    opx = np.array([cf.annual_opex_inflated.get(y, 0.0) for y in years])
    abx = np.array([cf.annual_abex_inflated.get(y, 0.0) for y in years])
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
    oil_prod = [cf.annual_oil_production.get(y, 0.0) for y in years]
    gas_prod = [cf.annual_gas_production.get(y, 0.0) for y in years]
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
    royalty = np.array([cf.annual_royalty.get(y, 0.0) for y in years])
    cap = np.array([cf.annual_capex_inflated.get(y, 0.0) for y in years])
    opx = np.array([cf.annual_opex_inflated.get(y, 0.0) for y in years])
    abx = np.array([cf.annual_abex_inflated.get(y, 0.0) for y in years])
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
    fig.add_trace(go.Bar(x=years, y=-royalty, name='Royalty', marker_color=color_palette[1]), row=2, col=1)
    fig.add_trace(go.Bar(x=years, y=-cap, name='CAPEX', marker_color=color_palette[2]), row=2, col=1)
    fig.add_trace(go.Bar(x=years, y=-opx, name='OPEX', marker_color=color_palette[3]), row=2, col=1)
    fig.add_trace(go.Bar(x=years, y=-abx, name='ABEX', marker_color=color_palette[4]), row=2, col=1)
    fig.add_trace(go.Bar(x=years, y=-tax, name='Tax', marker_color=color_palette[5]), row=2, col=1)
    fig.add_trace(go.Scatter(x=years, y=net, name='NCF', line=dict(color='black', width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=years, y=cum, name='Cumulative', line=dict(color='purple', dash='dot')), row=2, col=1, secondary_y=True)

    fig.update_layout(height=height, width=width, template='plotly_white', barmode='relative')
    return fig


def plot_cashflow(cf):
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
        rows=1, cols=1,
        subplot_titles=("Annual & Cumulative Cash Flow"),
        specs=[
            [{"type": "xy", "secondary_y": True}],
        ],
    )
    # Main Chart
    color_palette = px.colors.qualitative.Set3
    fig.add_trace(go.Bar(x=years, y=rev, name='Revenue', marker_color=color_palette[0]), row=1, col=1)
    fig.add_trace(go.Bar(x=years, y=-royalty, name='Royalty', marker_color=color_palette[1]), row=1, col=1)
    fig.add_trace(go.Bar(x=years, y=-cap, name='CAPEX', marker_color=color_palette[2]), row=1, col=1)
    fig.add_trace(go.Bar(x=years, y=-opx, name='OPEX', marker_color=color_palette[3]), row=1, col=1)
    fig.add_trace(go.Bar(x=years, y=-abx, name='ABEX', marker_color=color_palette[4]), row=1, col=1)
    fig.add_trace(go.Bar(x=years, y=-tax, name='Tax', marker_color=color_palette[5]), row=1, col=1)
    fig.add_trace(go.Scatter(x=years, y=net, name='NCF', line=dict(color='black', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=years, y=cum, name='Cumulative', line=dict(color='purple', dash='dot')), row=1, col=1, secondary_y=True)

    y_max = max(max(rev), max(cap+royalty+opx+abx+tax)) * 1.1
    y2_max = max(abs(cum)) * 1.1

    fig.update_layout(
        template='plotly_white', 
        barmode='relative',
        yaxis=dict(range=[-y_max, y_max]),
        yaxis2=dict(range=[-y2_max, y2_max], overlaying='y', side='right')
    )
    return fig


# Aggregate values for the entire project
def plot_cf_waterfall_chart(cf, width=1200,height=600):
    rev_oil = sum(cf.annual_revenue_oil.values())
    rev_gas = sum(cf.annual_revenue_gas.values())
    revenue = sum(cf.annual_revenue.values())
    capex = sum(cf.annual_capex.values())
    opex = sum(cf.annual_opex.values())
    abex = sum(cf.annual_abex.values())
    royalty = sum(cf.annual_royalty.values())
    income_tax = sum(cf.annual_total_tax.values())
    ncf = sum(cf.annual_net_cash_flow.values())

    # make_subplot 명령은 디폴트로 'xy' 타입 서프플롯을 만듬
    # pie chart를 만들기 위해서 domain 형태의 플롯을 별도로 정의
    fig = make_subplots(rows=1, cols=2,
                        specs=[[{"type": "xy"}, {"type": "domain"}]],
                        column_widths=[0.6, 0.4],
                        subplot_titles=("Waterfall Chart", "Profit Distribution"))

    fig.update_layout(width=width,height=height,
                    template='plotly_white',    #  ['ggplot2', 'seaborn', 'simple_white', 'plotly', 'plotly_white', 'plotly_dark', 'presentation', 'xgridoff', 'ygridoff', 'gridon', 'none']
                    # title_text="Economic Analysis",
                    # title_x = 0.5,
                    # title_xanchor = 'center',
                    # title_font_size = 20,
    )

    # --- Define the Set3 Colors ---
    set3_colors = px.colors.qualitative.Set3
    INCREASING_COLOR = set3_colors[6] # rgb(179,222,105)
    DECREASING_COLOR = set3_colors[3] # rgb(251,128,114)
    TOTAL_COLOR      = set3_colors[5] # rgb(253,180,98)

    # Waterfall Chart 데이터
    x_waterfall = ['Oil Sales', 'Gas Sales', 'CAPEX', 'OPEX', 'ABEX', 'Royalty', 'Income Tax', 'Net Cash Flow']
    y_waterfall = [rev_oil, rev_gas, -capex, -opex, -abex, -royalty, -income_tax, ncf]
    measure = ["relative", "relative", "relative", "relative", "relative", "relative", "relative", "total"]

    # x_waterfall = ['Revenue', 'CAPEX', 'OPEX', 'ABEX', 'Royalty', 'Income Tax', 'Net Cash Flow']
    # y_waterfall = [revenue, -capex, -opex, -abex, -royalty, -income_tax, ncf]
    # measure = ["relative", "relative", "relative", "relative", "relative", "relative", "total"]



    label_text = []
    for value in y_waterfall:
        converted_text = f"{abs(value):,.0f}MM$"
        label_text.append(converted_text)

    # 1. Waterfall Chart-------------------------
    fig.add_trace(go.Waterfall(
        name = "Cash Flow Distribution",
        orientation = "v",
        x = x_waterfall,
        y = y_waterfall,
        measure = measure,
        textposition = "outside",
        text = label_text,
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
        increasing = {"marker":{"color": INCREASING_COLOR}},
        decreasing = {"marker": {"color": DECREASING_COLOR}},
        totals =  {"marker":{"color": TOTAL_COLOR}},
        ), row=1, col=1)

    # Pie Chart 데이터
    labels = ['CAPEX', 'OPEX', 'ABEX', 'Royalty', 'Income Tax', 'Net Cash Flow']
    values = [capex, opex, abex, royalty, income_tax, ncf ]
    pull_apart = [0, 0, 0, 0, 0, 0.2]

    # 2. Pie Chart-------------------------
    fig.add_trace(go.Pie(
        labels=labels,
        values=values,
        textinfo='label+percent',
        textposition = "inside",
        marker_colors=px.colors.qualitative.Set3, #px.colors.sequential.RdBu,
        hole=.3,
        pull=pull_apart,
        sort=False, # 기본값은 값의 크기로 정렬되어 그려짐. 리스트의 오더대로 파이가 그려지도록 설정
        ), row=1, col=2)

    fig.update_layout(height=height, width=width, template='plotly_white', barmode='relative')
    return fig

def plot_cf_sankey_chart(cf, width=800, height=600):
    # --- 1. 데이터 추출 및 계산 ---
    rev_oil = sum(cf.annual_revenue_oil.values())
    rev_gas = sum(cf.annual_revenue_gas.values())
    revenue = sum(cf.annual_revenue.values())
    capex = sum(cf.annual_capex.values())
    opex = sum(cf.annual_opex.values())
    abex = sum(cf.annual_abex.values())
    royalty = sum(cf.annual_royalty.values())
    income_tax = sum(cf.annual_total_tax.values())
    ncf = sum(cf.annual_net_cash_flow.values())
    # 중간 노드별 합계 계산
    cost_total = capex + opex + abex
    gross_profit = revenue - cost_total
    gov_share_total = royalty + income_tax

    # --- 2. 색상 정의 ---
    # 원하는 투명도(Alpha) 설정 (예: 70% 투명도)
    OPACITY = 0.5
    colors = px.colors.qualitative.Pastel1
    Link_COLOR_NCF = get_rgba_from_hex(colors[0], OPACITY)
    Link_COLOR_GOV_SHARE = get_rgba_from_hex(colors[2], OPACITY)
    Link_COLOR_COSTS = get_rgba_from_hex(colors[1], OPACITY)
    COLOR_NCF = colors[0]
    COLOR_GOV_SHARE = colors[2]
    COLOR_COSTS = colors[1]

    # --- 3. 노드 정의 및 값 포맷팅 ---
    # 노드 이름 정의
    node_names = [
        'Oil Sales', 'Gas Sales', 'Revenue',
        'Gross Profit',  'Total Costs',
        'Net Profit', 'Costs', 'Goverment',
        'Net Cash Flow', 'CAPEX', 'OPEX', 'ABEX', 'Royalty', 'Income Tax',
    ]
    # 노드 값 정의 (노드 이름 순서와 일치)
    node_values = [
        rev_oil, rev_gas, revenue,
        gross_profit, cost_total,
        ncf, cost_total, gov_share_total,        # Goverment
        ncf, capex, opex, abex, royalty, income_tax,
    ]

    # 비율 계산 및 라벨 포맷팅: Label <br> Value <br> Ratio%
    nodes_with_formatted_labels = []
    def calculate_ratio_and_format_label(name, value, total_rev):
        ratio = (value / total_rev) * 100 if total_rev else 0
        value_str = f"{value:,.0f}MM$"
        ratio_str = f"({ratio:.1f}%)"
        # HTML <br> 태그를 사용하여 줄 바꿈
        return f"{name}<br>{value_str}<br>{ratio_str}"

    for name, value in zip(node_names, node_values):
        formatted_label = calculate_ratio_and_format_label(name, value, revenue)
        nodes_with_formatted_labels.append(formatted_label)

    nodes = nodes_with_formatted_labels
    node_map = {name: i for i, name in enumerate(node_names)}

    # --- 4. Sankey Diagram 링크 정의 및 값 포맷팅 ---
    links = []

    # 포맷팅 함수 정의
    def format_link_label(name, value):
        return f'{name}: {value:,.0f}MM$'

    # 0단계 분기: Division -> Revenue (0)
    links.append({'source': node_map['Oil Sales'],
                  'target': node_map['Revenue'], 'value': rev_oil,
                  'label': format_link_label('Revenue', revenue), 'color': Link_COLOR_NCF})
    links.append({'source': node_map['Gas Sales'],
                  'target': node_map['Revenue'], 'value': rev_gas,
                  'label': format_link_label('Revenue', revenue), 'color': Link_COLOR_NCF})

    # 1단계 분기: Revenue (0) -> Gross Profit (1), Costs (1)
    links.append({'source': node_map['Revenue'],
                  'target': node_map['Gross Profit'], 'value': gross_profit,
                  'label': format_link_label('Gross Profit', gross_profit), 'color': Link_COLOR_NCF})
    links.append({'source': node_map['Revenue'],
                  'target': node_map['Total Costs'], 'value': cost_total,
                  'label': format_link_label('Total Costs', cost_total), 'color': Link_COLOR_COSTS})

    # 2단계 분기: Gross Profit (1) -> 중간 분류 노드 (Net Profit, Gov. Share)
    links.append({'source': node_map['Gross Profit'],
                  'target': node_map['Net Profit'], 'value': ncf,
                  'label': format_link_label('NCF', ncf), 'color': Link_COLOR_NCF})
    links.append({'source': node_map['Gross Profit'],
                  'target': node_map['Goverment'], 'value': gov_share_total,
                  'label': format_link_label('Goverment', gov_share_total), 'color': Link_COLOR_GOV_SHARE})
    # 3단계 분기: Cost (1) -> Costs (2)
    links.append({'source': node_map['Total Costs'],
                  'target': node_map['Costs'], 'value': cost_total,
                  'label': format_link_label('Costs', cost_total), 'color': Link_COLOR_COSTS})

    # 2단계 분기: 중간 분류 노드 -> 최종 항목 노드
    # Profit -> NCF
    links.append({'source': node_map['Net Profit'],
                  'target': node_map['Net Cash Flow'], 'value': ncf,
                  'label': format_link_label('NCF', ncf), 'color': Link_COLOR_NCF})

    # Goverment -> Royalty, Income Tax
    links.append({'source': node_map['Goverment'],
                  'target': node_map['Royalty'], 'value': royalty,
                  'label': format_link_label('Royalty', royalty), 'color': Link_COLOR_GOV_SHARE})
    links.append({'source': node_map['Goverment'],
                  'target': node_map['Income Tax'], 'value': income_tax,
                  'label': format_link_label('Income Tax', income_tax), 'color': Link_COLOR_GOV_SHARE})

    # Costs -> CAPEX, OPEX, ABEX
    links.append({'source': node_map['Costs'],
                  'target': node_map['CAPEX'], 'value': capex,
                  'label': format_link_label('CAPEX', capex), 'color': Link_COLOR_COSTS})
    links.append({'source': node_map['Costs'],
                  'target': node_map['OPEX'], 'value': opex,
                  'label': format_link_label('OPEX', opex), 'color': Link_COLOR_COSTS})
    links.append({'source': node_map['Costs'],
                  'target': node_map['ABEX'], 'value': abex,
                  'label': format_link_label('ABEX', abex), 'color': Link_COLOR_COSTS})

    df_links = pd.DataFrame(links)

    # --- 5. Sankey Chart 생성 ---
    fig = go.Figure(data=[go.Sankey(
        # 노드 설정
        node=dict(
            pad=20,
            thickness=10,
            line=dict(width=0), # 테두리 제거
            # line=dict(color="black", width=0.2), # 테두리
            label=nodes, # 노드 라벨에 금액 포함
            # 노드 색상 지정
            color= [COLOR_NCF]*3 + [COLOR_NCF] + [COLOR_COSTS]
             + [COLOR_NCF] + [COLOR_COSTS] + [COLOR_GOV_SHARE]
             + [COLOR_NCF] + [COLOR_COSTS] * 3 + [COLOR_GOV_SHARE] * 2,
            ),
        # 링크 설정
        link=dict(
            source=df_links['source'],
            target=df_links['target'],
            value=df_links['value'],
            color=df_links['color'],
            label=df_links['label'], # 링크 라벨(hover text)에 금액 포함
        )
    )])

    # 차트 레이아웃 설정
    fig.update_layout(
        title_text="Cash Flow - Sankey Diagram",
        title_x=0.5,
        title_xanchor='center',
        title_font_size=20,
        font_size=12,
        width=width,
        height=height,
        template='plotly_white'
    )

    return fig