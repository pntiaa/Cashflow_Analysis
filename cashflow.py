# from pandas.core import base
import pandas as pd
from typing import Dict, Callable, Optional, Union, List
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

#---------------------------
# Cash Flow
# ---------------------------
class CashFlow_KOR_Regime:
    def __init__(self,
                 base_year: int = 2024,
                 cost_inflation_rate: float = 0.02,
                 discount_rate: float = 0.10,
                 exchange_rate: float = 1350.0,
                 oil_price_by_year: Dict[int, float] = {},
                 gas_price_by_year: Dict[int, float] = {},
                 oil_production_by_year: Dict[int, float] = {},
                 gas_production_by_year: Dict[int, float] = {}
                 ):
        # price / macro
        self.base_year = base_year
        self.cost_inflation_rate = cost_inflation_rate
        self.discount_rate = discount_rate
        self.exchange_rate = exchange_rate  # KRW / USD
        self.oil_price_by_year = oil_price_by_year
        self.gas_price_by_year = gas_price_by_year

        # tax params (simple)
        # self.local_income_tax_rate = 0.0  # 10% of corporate tax by default
        self.development_cost = None

        # convenience copies
        self.dev_cost_years: List[int] = []
        self.dev_annual_capex: Dict[int, float] = {}
        self.dev_annual_opex: Dict[int, float] = {}
        self.dev_annual_abex: Dict[int, float] = {}

        self.dev_annual_capex_inflated: Dict[int, float] = {}
        self.dev_annual_opex_inflated: Dict[int, float] = {}
        self.dev_annual_abex_inflated: Dict[int, float] = {}

        self.dev_capex_breakdown: Dict[str, Dict[int, float]] = {}

        self.annual_cum_revenue: Dict[str, Dict[int, float]] = {}
        self.annual_cum_capex: Dict[str, Dict[int, float]] = {}
        self.annual_cum_opex: Dict[str, Dict[int, float]] = {}
        self.annual_r_factor: Dict[str, Dict[int, float]] = {}
        self.annual_royalty : Dict[str, Dict[int, float]] = {}

        # production (two accepted forms)
        # if production provided as sequence: use arrays + production_start_year
        # if provided as dict: we will use dict form directly
        self.oil_production_series = None  # if array-like, units bbl/year
        self.gas_production_series = None  # if array-like, units mcf/year
        self.production_start_year: Optional[int] = None
        self.production_years: Optional[int] = None

        # Depreciation (dict by year)
        self.annual_depreciation: Dict[int, float] = {}

        # Annual accounting dicts
        self.annual_revenue: Dict[int, float] = {}
        self.annual_revenue_oil: Dict[int, float] = {}
        self.annual_revenue_gas: Dict[int, float] = {}
        self.annual_capex: Dict[int, float] = {}
        self.annual_opex: Dict[int, float] = {}
        self.annual_abex: Dict[int, float] = {}
        self.taxable_income: Dict[int, float] = {}
        self.corporate_income_tax: Dict[int, float] = {}
        # self.local_income_tax: Dict[int, float] = {}
        self.annual_total_tax: Dict[int, float] = {}
        self.annual_net_cash_flow: Dict[int, float] = {}
        self.cumulative_cash_flow: Dict[int, float] = {}

        # Total Value
        self.irr: float = None
        self.payback: int = None
        self.total_revenue: float = None
        self.total_royalty: float = None
        self.total_capex: float = None
        self.total_opex: float = None
        self.total_abex: float = None

        # timeline
        self.all_years: List[int] = []
        self.project_years: int = 0

        # NPV / IRR
        self.npv: Optional[float] = None
        self.present_values: Dict[int, float] = {}

        # prints
        # print(f"CashFlow initialized: base_year={self.base_year}, "
        #       f"discount_rate={self.discount_rate}, exchange_rate={self.exchange_rate}")

    # ---------------------------
    # Helpers
    # ---------------------------
    @staticmethod
    def _ensure_years_union(*dicts) -> List[int]:
        years = set()
        for d in dicts:
            if d is None:
                continue
            years |= set(d.keys())
        return sorted(years)

    @staticmethod
    def _zero_dict_for_years(years: List[int]) -> Dict[int, float]:
        return {y: 0.0 for y in years}

    # ---------------------------
    # Set development cost object
    # ---------------------------
    def set_development_costs(self, dev, output=True):
        """
        Accept a DevelopmentCost instance (dict-based) or a dict with equivalent keys.
        Expected attributes/keys: cost_years (list), annual_capex (dict), annual_opex (dict),
        annual_abex (dict), capex_breakdown (optional).
        """
        if dev is None:
            raise ValueError("development cost must be provided")

        self.development_cost = dev

        # If dev is object with attributes
        if hasattr(dev, 'cost_years'):
            self.dev_cost_years = list(getattr(dev, 'cost_years', []))
            self.dev_annual_capex = dict(getattr(dev, 'annual_capex', {}))
            self.dev_annual_opex = dict(getattr(dev, 'annual_opex', {}))
            self.dev_annual_abex = dict(getattr(dev, 'annual_abex', {}))
            # capex breakdown if available (dictionary of dicts)
            capex_breakdown = getattr(dev, 'get_cost_breakdown', None)
            # prefer to use get_cost_breakdown if present
            if callable(capex_breakdown):
                breakdown = dev.get_cost_breakdown()
                self.dev_capex_breakdown = breakdown.get('capex_breakdown', {})
            else:
                # try direct attribute
                self.dev_capex_breakdown = dict(getattr(dev, 'capex_breakdown', {}))
        elif isinstance(dev, dict):
            self.dev_cost_years = list(dev.get('cost_years', []))
            self.dev_annual_capex = dict(dev.get('annual_capex', {}))
            self.dev_annual_opex = dict(dev.get('annual_opex', {}))
            self.dev_annual_abex = dict(dev.get('annual_abex', {}))
            self.dev_capex_breakdown = dict(dev.get('capex_breakdown', {}))
        else:
            raise ValueError("Unsupported development_cost type")

        # Normalize keys: ensure ints and sorted lists
        self.dev_cost_years = sorted([int(y) for y in self.dev_cost_years])
        self.dev_annual_capex = {int(k): float(v) for k, v in self.dev_annual_capex.items()}
        self.dev_annual_opex = {int(k): float(v) for k, v in self.dev_annual_opex.items()}
        self.dev_annual_abex = {int(k): float(v) for k, v in self.dev_annual_abex.items()}

        if output:
            print(f"[set_development_costs] dev years: {self.dev_cost_years}")
            print(f"  total_capex (sum): {sum(self.dev_annual_capex.values()):.3f} MM")
            print(f"  total_opex (sum): {sum(self.dev_annual_opex.values()):.3f} MM")
            print(f"  total_abex (sum): {sum(self.dev_annual_abex.values()):.3f} MM")

    # ---------------------------
    # Production setters
    # ---------------------------
    def set_production_profile_from_arrays(self, oil_prod, gas_prod, production_start_year: int):
        """
        Accept arrays/lists for oil and gas (annual volumes) and a production_start_year.
        These will be converted to dicts {year: volume}.
        """
        oil_arr = np.array(oil_prod)
        gas_arr = np.array(gas_prod)
        n = len(oil_arr)
        if len(gas_arr) != n:
            raise ValueError("oil and gas production arrays must have same length")
        self.production_start_year = int(production_start_year)
        self.production_years = n
        self.oil_production_by_year = {self.production_start_year + i: float(oil_arr[i]) for i in range(n)}
        self.gas_production_by_year = {self.production_start_year + i: float(gas_arr[i]) for i in range(n)}
        # print(f"[production] set from arrays: start={self.production_start_year}, years={self.production_years}")

    def set_production_profile_from_dicts(self, oil_dict: Dict[int, float], gas_dict: Dict[int, float]):
        """
        Accept dicts {year: volume}. Sets production_start_year and production_years accordingly.
        """
        self.oil_production_by_year = {int(k): float(v) for k, v in oil_dict.items()}
        self.gas_production_by_year = {int(k): float(v) for k, v in gas_dict.items()}
        years = sorted(set(self.oil_production_by_year.keys()) | set(self.gas_production_by_year.keys()))
        if len(years) == 0:
            raise ValueError("empty production dicts")
        self.production_start_year = years[0]
        self.production_years = len(years)
        # print(f"[production] set from dicts: start={self.production_start_year}, years={self.production_years}")

    # ---------------------------
    # Depreciation
    # ---------------------------
    def calculate_depreciation(self, method: str = 'straight_line',
                               useful_life: int = 10,
                               depreciable_components: Optional[List[str]] = None,
                               output=True):
        """
        Build annual_depreciation dict across full project timeline.
        depreciable_components: list of keys that exist in dev_capex_breakdown dict (e.g., ['drilling','subsea','FPSO',...])
        If None, take drilling+subsea+FPSO+export_pipeline+terminal+PM_others if available.
        """
        if self.development_cost is None:
            raise ValueError("Set development costs first")

        # select depreciable total (scalar in MM$)
        if depreciable_components is None:
            # try to sum common keys if breakdown exists
            try:
                breakdown = self.dev_capex_breakdown
                keys = ['drilling', 'subsea', 'FPSO', 'export_pipeline', 'terminal', 'PM_others']
                total_depr = 0.0
                for k in keys:
                    # breakdown[k] may be dict year->value; sum if dict
                    val = breakdown.get(k)
                    if isinstance(val, dict):
                        total_depr += sum(val.values())
                    elif isinstance(val, (int, float)):
                        total_depr += float(val)
            except Exception:
                total_depr = sum(self.dev_annual_capex.values())
        else:
            total_depr = 0.0
            for k in depreciable_components:
                val = self.dev_capex_breakdown.get(k, {})
                if isinstance(val, dict):
                    total_depr += sum(val.values())
                elif isinstance(val, (int, float)):
                    total_depr += float(val)

        # Build full timeline years first
        self._build_full_timeline()
        years = self.all_years

        # initialize years
        self.annual_depreciation = {y: 0.0 for y in years}

        if method == 'straight_line':
            ann = total_depr / float(useful_life) if useful_life > 0 else 0.0
            for i, y in enumerate(years):
                if i < useful_life:
                    self.annual_depreciation[y] = ann
                else:
                    self.annual_depreciation[y] = 0.0
        elif method == 'declining_balance':
            rate = 2.0 / float(useful_life) if useful_life > 0 else 0.0
            remaining = total_depr
            for i, y in enumerate(years):
                if i < useful_life and remaining > 0:
                    dep = remaining * rate
                    self.annual_depreciation[y] = dep
                    remaining -= dep
                else:
                    self.annual_depreciation[y] = 0.0
        else:
            raise ValueError("Unknown depreciation method")

        if output:
            print(f"[depr] method={method}, total_depr={total_depr:.3f} MM, useful_life={useful_life}")
            return self.annual_depreciation

    # ---------------------------
    # Build timeline helper
    # ---------------------------
    def _build_full_timeline(self):
        """
        Construct self.all_years (sorted) covering development years and production years and
        populate self.annual_capex/opex/abex by aligning dev dicts and filling zeros where missing.
        """
        if self.development_cost is None:
            raise ValueError("Set development costs first")

        # union of years
        # exploration 기간 포함필요====================================================================================check
        years = set(self.dev_cost_years)
        years |= set(self.oil_production_by_year.keys())
        years |= set(self.gas_production_by_year.keys())
        # If dev included an extra year (e.g., year_0), it is already in dev_cost_years
        # Abandonment 기간은 ?? 별도로 포함필요?==========================================================================check

        if len(years) == 0:
            raise ValueError("No years found (development or production)")

        self.all_years = sorted(list(years))
        self.project_years = len(self.all_years)

        # build annual series by year, default 0.0
        self.annual_capex = {y: float(self.dev_annual_capex.get(y, 0.0)) for y in self.all_years}
        self.annual_opex = {y: float(self.dev_annual_opex.get(y, 0.0)) for y in self.all_years}
        self.annual_abex = {y: float(self.dev_annual_abex.get(y, 0.0)) for y in self.all_years}

        # print(f"[timeline] all_years: {self.all_years}")

    # ---------------------------
    # Revenue calculation
    # ---------------------------
    def calculate_annual_revenue(self, output=True):
        """
        Compute annual revenue dict {year: MM$}
        revenue = oil_vol(MMbbl) * oil_price($/bbl) + gas_vol(bcf) * gas_price($/mcf)
        """
        if not self.oil_production_by_year and not self.gas_production_by_year:
            raise ValueError("Production profile not set")

        # **필수: 전체 프로젝트 연도 생성**
        self._build_full_timeline()

        years = sorted(set(self.oil_production_by_year.keys()) | set(self.gas_production_by_year.keys()))
        rev = {};  rev_oil = {}; rev_gas = {}
        for y in years:
            oil_vol = self.oil_production_by_year.get(y, 0.0)
            gas_vol = self.gas_production_by_year.get(y, 0.0)
            oil_price = self.oil_price_by_year.get(y, 0.0)
            gas_price = self.gas_price_by_year.get(y, 0.0)
            revenue_oil = (oil_vol * oil_price)  # oil volume in MMbbls and gas volume in bcf
            revenue_gas = (gas_vol * gas_price)  # oil volume in MMbbls and gas volume in bcf
            revenue =revenue_oil + revenue_gas
            rev_oil[y] = float(revenue_oil)
            rev_gas[y] = float(revenue_gas)
            rev[y] = float(revenue)

        # populate for full timeline (zeros where no production)
        self.annual_revenue = {y: float(rev.get(y, 0.0)) for y in self.all_years}
        self.annual_revenue_oil = {y: float(rev_oil.get(y, 0.0)) for y in self.all_years}
        self.annual_revenue_gas = {y: float(rev_gas.get(y, 0.0)) for y in self.all_years}

        if output:
            print(f"[revenue] total revenue (MM$): {sum(self.annual_revenue.values()):.3f}")
            return self.annual_revenue



    # ---------------------------
    # Taxes
    # ---------------------------
    def _calculate_CIT(self, taxable_income: float)->float:
        '''
        법인세율 계산함수.
        MM$ 단위를 천만원으로 변환하여 계산후, 다시 MM$로 변환
        '''
        taxable_income_krw = taxable_income * self.exchange_rate / 10
        if taxable_income_krw > 30_000:
            CIT_krw = (6268 + ( taxable_income_krw - 30_000) * 0.24) * 1.1
            CIT = CIT_krw / self.exchange_rate * 10 # convert to mil USD
        elif taxable_income_krw > 2_000:
            CIT_krw = (378 + ( taxable_income_krw - 2_000) * 0.21) * 1.1
            CIT = CIT_krw / self.exchange_rate * 10
        elif taxable_income_krw > 20:
            CIT_krw = (1.8 + ( taxable_income_krw - 2) * 0.19) * 1.1
            CIT = CIT_krw / self.exchange_rate * 10
        elif taxable_income_krw > 0:
            CIT_krw = (taxable_income_krw * 0.09) * 1.1
            CIT = CIT_krw / self.exchange_rate * 10
        else:
            CIT = 0
        return CIT

    def calculate_taxes(self, output=True):
        """
        Compute taxable income, corporate tax, local tax, total tax per year.
        Taxable income = revenue - (opex + abex) - depreciation
        Taxes calculated only where taxable_income > 0
        """
        # Build full timeline
        self._build_full_timeline()

        # Ensure revenue exists
        if not self.annual_revenue:
            self.calculate_annual_revenue()

        # Ensure depreciation exists
        if not self.annual_depreciation:
            # default depreciation: straight line 10 years
            self.calculate_depreciation()

        self.taxable_income = {}
        self.corporate_income_tax = {}
        # self.local_income_tax = {}
        self.annual_total_tax = {}
        self.total_tax:float = None

        for y in self.all_years:
            rev = float(self.annual_revenue.get(y, 0.0))
            opex = float(self.annual_opex.get(y, 0.0))
            abex = float(self.annual_abex.get(y, 0.0))
            depr = float(self.annual_depreciation.get(y, 0.0))
            taxable = rev - (opex + abex) - depr
            # only positive taxable income is taxed
            corp_tax = self._calculate_CIT(taxable)
            # local_tax = corp_tax * self.local_income_tax_rate
            # total = corp_tax + local_tax
            total = corp_tax
            self.taxable_income[y] = taxable
            self.corporate_income_tax[y] = corp_tax
            # self.local_income_tax[y] = local_tax
            self.annual_total_tax[y] = total

        self.total_tax = sum(self.annual_total_tax.values())
        if output:
            print(f"[taxes] total tax (MM$): {sum(self.annual_total_tax.values()):.3f}"
            f"(KRW: {sum(self.annual_total_tax.values()) * self.exchange_rate:.0f})")
            return self.annual_total_tax,


    def _calculate_royalty_rates(self, r_factor:float):
        if r_factor < 1.25:
            royalty_rates = 0.01
        elif r_factor <3:
            royalty_rates = round(((18.28 * (r_factor-1.25)) + 1)/100,2)
        else:
            royalty_rates = 0.33
        return royalty_rates

    def calculate_royalty(self):
        # Ensure revenue computed
        if not self.annual_revenue:
            self.calculate_annual_revenue()

        # Build full timeline
        self._build_full_timeline()

        # Ensure revenue exists
        if not self.annual_revenue:
            self.calculate_annual_revenue()

        years = list(self.annual_capex.keys())

        # 해당년까지의 누적 Capex 계산
        cum_revenue = np.cumsum([self.annual_revenue.get(y, 0.0) for y in years])
        cum_capex = np.cumsum([self.annual_capex.get(y, 0.0) for y in years])
        cum_opex = np.cumsum([self.annual_opex.get(y, 0.0) for y in years])
        annual_cum_revenue = {}
        annual_cum_capex = {}
        annual_cum_opex = {}
        for i, y in enumerate(years):
            annual_cum_revenue[y] =cum_revenue[i]
            annual_cum_capex[y] =cum_capex[i]
            annual_cum_opex[y] =cum_opex[i]

        self.annual_cum_revenue = annual_cum_revenue
        self.annual_cum_capex = annual_cum_capex
        self.annual_cum_opex = annual_cum_opex

        annual_r_factor = {}
        annual_royalty_rates = {}
        annual_royalty = {}
        cum_revenue_after_royalty = {}
        cum_royalty = 0

        for y in years:
            cum_revenue_after_royalty[y] = self.annual_cum_revenue.get(y, 0.0) - cum_royalty
            annual_r_factor[y] = cum_revenue_after_royalty[y] / (annual_cum_capex[y]+annual_cum_opex[y])
            annual_royalty_rates[y] = self._calculate_royalty_rates(annual_r_factor[y])
            # '조광료'는 부과대상연도의 매출액에 조광료 부과요율을 곱하여 산정한다.
            annual_royalty[y] = self.annual_revenue[y] * annual_royalty_rates[y]
            cum_royalty +=  annual_royalty[y]

        self.annual_r_factor = annual_r_factor
        self.annual_royalty_rates = annual_royalty_rates
        self.cum_revenue_after_royalty = cum_revenue_after_royalty
        self.annual_royalty = annual_royalty

        # return self.annual_royalty_rates, self.annual_royalty
        return None

    # ---------------------------
    # Net cash flow (after tax)
    # ---------------------------
    def calculate_net_cash_flow(self, output=True):
        """
        Combine revenue, capex, opex, abex, taxes to compute annual net cash flow and cumulative.
        Formula (after tax):
          net_cash_flow = revenue - (capex + opex + abex) - total_tax
        All per-year in MM$
        """
        if self.development_cost is None:
            raise ValueError("Development costs not set")

        # Build timeline and align series
        self._build_full_timeline()

        # Ensure revenue & taxes computed
        if not self.annual_revenue:
            self.calculate_annual_revenue(output=output)
        if not self.annual_total_tax:
            self.calculate_taxes(output=output)
        if not self.annual_royalty:
            self.calculate_royalty()

        self.annual_net_cash_flow = {}
        self.cumulative_cash_flow = {}
        cum = 0.0

        for y in self.all_years:
            rev = float(self.annual_revenue.get(y, 0.0))
            royalty = float(self.annual_royalty.get(y,0.0)) # royalty add
            capex = float(self.annual_capex.get(y, 0.0))
            opex = float(self.annual_opex.get(y, 0.0))
            abex = float(self.annual_abex.get(y, 0.0))
            tax = float(self.annual_total_tax.get(y, 0.0))
            ncf = rev - royalty - (capex + opex + abex) - tax
            self.annual_net_cash_flow[y] = ncf
            cum += ncf
            self.cumulative_cash_flow[y] = cum

        if output:
            print(f"[ncf] project duration {self.all_years[0]} - {self.all_years[-1]}, "
                f"final cumulative: {cum:.3f} MM")
            return self.annual_net_cash_flow

    # ---------------------------
    # NPV and IRR
    # ---------------------------
    def calculate_npv(self, discount_rate: Optional[float] = None, output=True):
        if self.annual_net_cash_flow is None or len(self.annual_net_cash_flow) == 0:
            self.calculate_net_cash_flow()
        if discount_rate is not None:
            self.discount_rate = discount_rate

        years = np.array(self.all_years)
        years_from_start = years - years[0]
        dfs = 1.0 / ((1.0 + self.discount_rate) ** years_from_start)

        pv = {}
        total_pv = 0.0
        for i, y in enumerate(self.all_years):
            pv[y] = float(self.annual_net_cash_flow.get(y, 0.0)) * float(dfs[i])
            total_pv += pv[y]

        self.present_values = pv
        self.npv = total_pv
        if output:
            print(f"[npv] discount_rate={self.discount_rate:.3f}, NPV={self.npv:.3f} MM")
            return self.npv

    def calculate_irr(self, max_iter: int = 200, tol: float = 1e-6):
        """
        Compute IRR by Newton-Raphson on the annual_net_cash_flow series aligned to self.all_years.
        Returns IRR as decimal (e.g., 0.12 for 12%), or None if not converged.
        """
        if self.annual_net_cash_flow is None or len(self.annual_net_cash_flow) == 0:
            return None

        # build arrays
        years = np.array(self.all_years)
        cf_array = np.array([self.annual_net_cash_flow[y] for y in self.all_years])

        # initial guess
        rate = 0.10
        for _ in range(max_iter):
            dfs = 1.0 / ((1.0 + rate) ** (years - years[0]))
            npv = np.sum(cf_array * dfs)
            # derivative dNPV/dr = sum(-t * cf_t / (1+r)^(t+1))
            t = (years - years[0]).astype(float)
            derivative = np.sum(-t * cf_array * dfs / (1.0 + rate))
            if abs(derivative) < 1e-12:
                break
            new_rate = rate - npv / derivative
            if abs(new_rate - rate) < tol:
                return new_rate
            rate = new_rate
        return None

    # ---------------------------
    # Plotting results
    # ---------------------------
    def plot_cash_flow_profile(self, show: bool = True, figsize:tuple = (16,12)):
        """
        Plots:
         - annual stacked bars: revenue (positive), costs & taxes (negative)
         - net cash flow line
         - cumulative cash flow subplot
         - tax & depreciation subplot
        """
        if not self.annual_net_cash_flow:
            self.calculate_net_cash_flow()

        years = self.all_years
        rev = np.array([self.annual_revenue.get(y, 0.0) for y in years])
        royalty = np.array([self.annual_royalty.get(y, 0.0) for y in years])
        cap = np.array([self.annual_capex.get(y, 0.0) for y in years])
        opx = np.array([self.annual_opex.get(y, 0.0) for y in years])
        abx = np.array([self.annual_abex.get(y, 0.0) for y in years])
        tax = np.array([self.annual_total_tax.get(y, 0.0) for y in years])
        net = np.array([self.annual_net_cash_flow.get(y, 0.0) for y in years])
        cum = np.array([self.cumulative_cash_flow.get(y, 0.0) for y in years])
        depr = np.array([self.annual_depreciation.get(y, 0.0) for y in years])
        corp = np.array([self.corporate_income_tax.get(y, 0.0) for y in years])
        local = np.array([self.local_income_tax.get(y, 0.0) for y in years])

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        ax1 = axes[0, 0]
        ax2 = axes[0, 1]
        ax3 = axes[1, 0]
        ax4 = axes[1, 1]

        # 1) Annual components
        ax1.bar(years, rev, label='Revenue', alpha=0.7)
        ax1.bar(years, -royalty,label='Royalty', alpha=0.7)
        ax1.bar(years, -(cap + opx + abx), bottom=-royalty, label='Costs (CAPEX+OPEX+ABEX)', alpha=0.7)
        ax1.bar(years, -tax, bottom=-(cap + opx + abx + royalty), label='Taxes', alpha=0.7)
        ax1.plot(years, net, 'o-', color='black', linewidth=2, label='Net Cash Flow (After Tax)')
        ax1.axhline(0.0, color='gray', linewidth=0.8)
        ax1.set_title('Annual Cash Flow Components')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('MM$')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2) Cumulative cash flow
        ax2.plot(years, cum, 'o-', color='purple', linewidth=2)
        ax2.fill_between(years, cum, alpha=0.2, color='purple')
        ax2.axhline(0.0, color='red', linestyle='--')
        # payback year
        payback_year = None
        for i, v in enumerate(cum):
            if v >= 0:
                payback_year = years[i]
                break
        if payback_year is not None:
            ax2.axvline(payback_year, color='green', linestyle='--')
            ax2.text(payback_year, cum[list(cum).index(next(filter(lambda x: x >= 0, cum)))],
                     f' Payback: {payback_year}', rotation=90, verticalalignment='bottom')
        ax2.set_title('Cumulative Cash Flow (After Tax)')
        ax2.set_xlabel('Year')
        ax2.set_ylabel('MM$')
        ax2.grid(True, alpha=0.3)

        # 3) Tax components & depreciation
        ax3.bar(years, corp, label='Corporate Tax', alpha=0.7)
        ax3.bar(years, local, bottom=corp, label='Local Tax', alpha=0.7)
        ax3.plot(years, depr, 'o-', color='gray', label='Depreciation')
        ax3.set_title('Taxes & Depreciation')
        ax3.set_xlabel('Year')
        ax3.set_ylabel('MM$')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4) Price profiles
        oilp = [self.oil_price_by_year[y] for y in years]
        gasp = [self.gas_price_by_year[y] for y in years]
        ax4.plot(years, oilp, 'o-', color='green', label='Oil Price ($/bbl)')
        ax4.set_title('Commodity Prices')
        ax4.set_xlabel('Year')
        # ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim([50,150])
        ax5 = ax4.twinx()
        ax5.plot(years, gasp, 'o-' ,color='red', label='Gas Price ($/mcf)')
        ax5.set_ylabel('Gas Price ($/mcf)')
        ax5.set_ylim([5,15])
        # ax5.legend()
        # combine legends for ax4 and ax5
        handles0, labels0 = ax4.get_legend_handles_labels()
        handles1, labels1 = ax5.get_legend_handles_labels()
        all_handles = handles0 + handles1
        all_labels = labels0 + labels1
        ax4.legend(all_handles, all_labels)

        plt.tight_layout()
        if show:
            plt.show()
    # ---------------------------
    # Summary
    # ---------------------------
    def get_project_summary(self):
        """
        Return key metrics dictionary (NPV, IRR, payback, totals).
        """
        if self.npv is None:
            self.calculate_npv()

        self.irr = self.calculate_irr()
        self.total_revenue = sum(self.annual_revenue.values()) if self.annual_revenue else 0.0
        self.total_royalty = sum(self.annual_royalty.values()) if self.annual_royalty else 0.0
        self.total_capex = sum(self.annual_capex.values()) if self.annual_capex else 0.0
        self.total_opex = sum(self.annual_opex.values()) if self.annual_opex else 0.0
        self.total_abex = sum(self.annual_abex.values()) if self.annual_abex else 0.0

        # payback
        for y, val in self.cumulative_cash_flow.items():
            if val >= 0:
                payback = y
                break
        self.payback = payback

        return {
            'total_revenue': self.total_revenue,
            'total_royalty': self.total_royalty,
            'total_capex': self.total_capex,
            'total_opex': self.total_opex,
            'total_abex': self.total_abex,
            'npv': self.npv,
            'irr': self.irr,
            'payback_year': self.payback,
           'final_cumulative': self.cumulative_cash_flow.get(self.all_years[-1], 0.0)
        }

if __name__ == "__main__":
    pass    