# from pandas.core import base
import pandas as pd
from typing import Dict, Callable, Optional, Union, List, Any
import numpy as np
import numpy_financial as npf
from pydantic import BaseModel, Field

#---------------------------
# Cash Flow
# ---------------------------

class CashFlowKOR(BaseModel):
    # 가격/거시 경제 지표
    base_year: int = 2024
    cost_inflation_rate: float = 0.02
    discount_rate: float = 0.10
    exchange_rate: float = 1350.0  # KRW / USD
    oil_price_by_year: Dict[int, float] = Field(default_factory=dict)
    gas_price_by_year: Dict[int, float] = Field(default_factory=dict)

    # 세금 관련 변수
    development_cost: Optional[Any] = None

    # 편의를 위한 개발 비용 정보
    cost_years: List[int] = Field(default_factory=list)
    annual_capex: Dict[int, float] = Field(default_factory=dict)
    annual_opex: Dict[int, float] = Field(default_factory=dict)
    annual_abex: Dict[int, float] = Field(default_factory=dict)
    annual_capex_inflated: Dict[int, float] = Field(default_factory=dict)
    annual_opex_inflated: Dict[int, float] = Field(default_factory=dict)
    annual_abex_inflated: Dict[int, float] = Field(default_factory=dict)
    capex_breakdown: Dict[str, Dict[int, float]] = Field(default_factory=dict)

    annual_cum_revenue: Dict[int, float] = Field(default_factory=dict)
    annual_cum_capex_inflated: Dict[int, float] = Field(default_factory=dict)
    annual_cum_opex_inflated: Dict[int, float] = Field(default_factory=dict)
    annual_cum_abex_inflated: Dict[int, float] = Field(default_factory=dict)
    annual_r_factor: Dict[int, float] = Field(default_factory=dict)
    annual_royalty: Dict[int, float] = Field(default_factory=dict)

    # 생산량
    oil_production_series: Optional[Any] = None
    gas_production_series: Optional[Any] = None
    production_start_year: Optional[int] = None
    production_years: Optional[int] = None
    annual_oil_production: Dict[int, float] = Field(default_factory=dict)
    annual_gas_production: Dict[int, float] = Field(default_factory=dict)
    total_oil_production: float = 0.0
    total_gas_production: float = 0.0

    # 감가상각 (연도별 딕셔너리)
    annual_depreciation: Dict[int, float] = Field(default_factory=dict)

    # 연간 회계 관련 딕셔너리
    discovery_bonus: Optional[float]= None
    annual_discovery_bonus: Dict[int, float] = Field(default_factory=dict) # 연도별
    annual_revenue: Dict[int, float] = Field(default_factory=dict)
    annual_revenue_oil: Dict[int, float] = Field(default_factory=dict)
    annual_revenue_gas: Dict[int, float] = Field(default_factory=dict)
    annual_royalty_rates: Dict[int, float] = Field(default_factory=dict)
    cum_revenue_after_royalty: Dict[int, float] = Field(default_factory=dict)
    annual_royalty: Dict[int, float] = Field(default_factory=dict)

    taxable_income: Dict[int, float] = Field(default_factory=dict)
    corporate_income_tax: Dict[int, float] = Field(default_factory=dict)
    other_fees: Dict[int, float] = Field(default_factory=dict)  # 공유수면점사용료, 교육훈련비
    annual_total_tax: Dict[int, float] = Field(default_factory=dict)
    annual_net_cash_flow: Dict[int, float] = Field(default_factory=dict)
    cumulative_cash_flow: Dict[int, float] = Field(default_factory=dict)

    # 총 가치
    cop_year: Optional[int] = None
    payback_year: Optional[int] = None
    total_revenue: Optional[float] = None
    total_royalty: Optional[float] = None
    total_capex: Optional[float] = None
    total_opex: Optional[float] = None
    total_abex: Optional[float] = None
    total_tax: Optional[float] = None

    # 타임라인
    all_years: List[int] = Field(default_factory=list)
    project_years: int = 0

    # NPV / IRR
    npv: Optional[float] = None
    irr: Optional[float] = None
    present_values: Dict[int, float] = Field(default_factory=dict)

    # ---------------------------
    # 헬퍼 함수
    # ---------------------------
    @staticmethod
    def _ensure_years_union(*dicts) -> List[int]:
        # 여러 딕셔너리의 연도를 통합하여 정렬된 리스트 반환
        years = set()
        for d in dicts:
            if d is None:
                continue
            years |= set(d.keys())
        return sorted(years)

    @staticmethod
    def _zero_dict_for_years(years: List[int]) -> Dict[int, float]:
        # 특정 연도 리스트에 대한 0 값 딕셔너리 생성
        return {y: 0.0 for y in years}

    @staticmethod
    def _sum_dict_values(d: Dict[int, float]) -> float:
        return sum(d.values())

    # 타임라인 구축 헬퍼 함수
    def _build_full_timeline(self):
        """
        - 개발 및 생산 연도를 포함하는 전체 연도(`self.all_years`)를 구축.
        - 개발 딕셔너리를 정렬하고 누락된 부분을 0으로 채워 `self.annual_capex/opex/abex`를 채움.
        """
        if self.development_cost is None:
            raise ValueError("개발 비용을 먼저 설정하십시오")

        # 연도 통합
        years = set(self.cost_years)
        years |= set(self.annual_oil_production.keys())
        years |= set(self.annual_gas_production.keys())
        years |= set(self.annual_capex.keys())
        years |= set(self.annual_opex.keys())
        years |= set(self.annual_abex.keys())

        if len(years) == 0:
            raise ValueError("연도를 찾을 수 없습니다 (개발 또는 생산)")

        self.all_years = sorted(list(years))
        self.project_years = len(self.all_years)

    # ---------------------------
    # 개발 비용 객체 설정
    # ---------------------------
    def set_development_costs(self, dev, output=True):
        """
        DevelopmentCost 인스턴스 또는 동등한 키를 가진 딕셔너리 수용.
        """
        if dev is None:
            raise ValueError("개발 비용이 제공되어야 합니다")

        self.development_cost = dev

        # 임시 변수에 dev 객체의 속성들을 복사하고 정규화
        temp_dev_cost_years = []
        temp_dev_annual_capex = {}
        temp_dev_annual_opex = {}
        temp_dev_annual_abex = {}
        temp_dev_capex_breakdown = {}

        if hasattr(dev, 'cost_years'):
            temp_dev_cost_years = list(getattr(dev, 'cost_years', []))
            temp_dev_annual_capex = dict(getattr(dev, 'annual_capex', {}))
            temp_dev_annual_opex = dict(getattr(dev, 'annual_opex', {}))
            temp_dev_annual_abex = dict(getattr(dev, 'annual_abex', {}))

            capex_breakdown_method = getattr(dev, 'get_cost_breakdown', None)
            if callable(capex_breakdown_method):
                breakdown_result = capex_breakdown_method()
                temp_dev_capex_breakdown = breakdown_result.get('capex_breakdown', {})
            else:
                temp_dev_capex_breakdown = dict(getattr(dev, 'capex_breakdown', {}))
        elif isinstance(dev, dict):
            temp_dev_cost_years = list(dev.get('cost_years', []))
            temp_dev_annual_capex = dict(dev.get('annual_capex', {}))
            temp_dev_annual_opex = dict(dev.get('annual_opex', {}))
            temp_dev_annual_abex = dict(dev.get('annual_abex', {}))
            temp_dev_capex_breakdown = dict(dev.get('capex_breakdown', {}))
        else:
            raise ValueError("지원되지 않는 development_cost 타입입니다")

        # 정규화된 키와 값을 self의 속성에 할당
        self.cost_years = sorted([int(y) for y in temp_dev_cost_years])
        self.annual_capex = {int(k): float(v) for k, v in temp_dev_annual_capex.items()}
        self.annual_opex = {int(k): float(v) for k, v in temp_dev_annual_opex.items()}
        self.annual_abex = {int(k): float(v) for k, v in temp_dev_annual_abex.items()}
        self.capex_breakdown = {k: {int(y): float(val) for y, val in v.items()} if isinstance(v, dict) else v for k, v in temp_dev_capex_breakdown.items()}

        # inflation 계산
        if self.cost_years:
            years = np.array(self.cost_years)
            years_from_start = years - years[0]
            inf = ((1.0 + self.cost_inflation_rate) ** years_from_start)
            for i, y in enumerate(years):
                self.annual_capex_inflated[y] = self.annual_capex.get(y, 0.0) * inf[i]
                self.annual_opex_inflated[y] = self.annual_opex.get(y, 0.0) * inf[i]
                self.annual_abex_inflated[y] = self.annual_abex.get(y, 0.0) * inf[i]

            cum_capex_inflated = np.cumsum([self.annual_capex_inflated.get(y, 0.0) for y in self.cost_years])
            cum_opex_inflated = np.cumsum([self.annual_opex_inflated.get(y, 0.0) for y in self.cost_years])
            cum_abex_inflated = np.cumsum([self.annual_abex_inflated.get(y, 0.0) for y in self.cost_years])

            for i, y in enumerate(self.cost_years):
                self.annual_cum_capex_inflated[y] =cum_capex_inflated[i]
                self.annual_cum_opex_inflated[y] =cum_opex_inflated[i]
                self.annual_cum_abex_inflated[y] =cum_abex_inflated[i]

        if output:
            print(f"[set_development_costs] 개발 연도: {self.cost_years}")
            print(f"  총 CAPEX (합계): {sum(self.annual_capex.values()):.3f} MM")

    # ---------------------------
    # 생산량 설정 함수
    # ---------------------------
    def set_production_profile_from_arrays(self, oil_prod, gas_prod, production_start_year: int):
        oil_arr = np.array(oil_prod)
        gas_arr = np.array(gas_prod)
        n = len(oil_arr)
        if len(gas_arr) != n:
            raise ValueError("석유 및 가스 생산량 배열의 길이가 동일해야 합니다")
        self.production_start_year = int(production_start_year)
        self.production_years = n
        self.annual_oil_production = {self.production_start_year + i: float(oil_arr[i]) for i in range(n)}
        self.annual_gas_production = {self.production_start_year + i: float(gas_arr[i]) for i in range(n)}
        self.total_gas_production = self._sum_dict_values(self.annual_gas_production)
        self.total_oil_production = self._sum_dict_values(self.annual_oil_production)

    def set_production_profile_from_dicts(self, oil_dict: Dict[int, float], gas_dict: Dict[int, float]):
        self.annual_oil_production = {int(k): float(v) for k, v in oil_dict.items()}
        self.annual_gas_production = {int(k): float(v) for k, v in gas_dict.items()}
        years = sorted(set(self.annual_oil_production.keys()) | set(self.annual_gas_production.keys()))
        if years:
            self.production_start_year = years[0]
            self.production_years = len(years)
            self.total_gas_production = self._sum_dict_values(self.annual_gas_production)
            self.total_oil_production = self._sum_dict_values(self.annual_oil_production)

    # ---------------------------
    # 감가상각
    # ---------------------------
    def calculate_depreciation(self, method: str = 'production_base', useful_life: int = 10, depreciable_components: Optional[List[str]] = None, output=True):
        if self.development_cost is None:
            raise ValueError("개발 비용을 먼저 설정하십시오")

        if depreciable_components is None:
            try:
                breakdown = self.capex_breakdown
                keys = ['drilling', 'subsea', 'FPSO', 'export_pipeline', 'terminal', 'PM_others']
                total_depr = 0.0
                for k in keys:
                    val = breakdown.get(k)
                    if isinstance(val, dict): total_depr += sum(val.values())
                    elif isinstance(val, (int, float)): total_depr += float(val)
            except Exception:
                total_depr = sum(self.annual_capex.values())
        else:
            total_depr = 0.0
            for k in depreciable_components:
                val = self.capex_breakdown.get(k, {})
                if isinstance(val, dict): total_depr += sum(val.values())
                elif isinstance(val, (int, float)): total_depr += float(val)

        self._build_full_timeline()
        years = self.all_years
        self.annual_depreciation = {y: 0.0 for y in years}
        remaining_reserve = self.total_gas_production

        if method == 'production_base':
            total_depr_amount = 0.0
            for y in years:
                gas_prod = self.annual_gas_production.get(y, 0.0)
                if remaining_reserve > 0 and gas_prod > 0:
                    current_cum_capex = self.annual_cum_capex_inflated.get(y, 0.0)
                    ratio = gas_prod / remaining_reserve
                    dep = (current_cum_capex - total_depr_amount) * ratio
                    self.annual_depreciation[y] = dep
                    total_depr_amount += dep
                    remaining_reserve -= gas_prod
                else: self.annual_depreciation[y] = 0.0
        elif method == 'straight_line':
            ann = total_depr / float(useful_life) if useful_life > 0 else 0.0
            for i, y in enumerate(years):
                if i < useful_life: self.annual_depreciation[y] = ann
                else: self.annual_depreciation[y] = 0.0
        return self.annual_depreciation

    # ---------------------------
    # 수익 계산
    # ---------------------------
    def calculate_annual_revenue(self, output=True):
        if not self.annual_oil_production and not self.annual_gas_production:
            raise ValueError("생산 프로필이 설정되지 않았습니다")
        self._build_full_timeline()
        rev = {y: 0.0 for y in self.all_years}
        rev_oil = {y: 0.0 for y in self.all_years}
        rev_gas = {y: 0.0 for y in self.all_years}
        for y in self.all_years:
            oil_vol = self.annual_oil_production.get(y, 0.0)
            gas_vol = self.annual_gas_production.get(y, 0.0)
            oil_price = self.oil_price_by_year.get(y, 0.0)
            gas_price = self.gas_price_by_year.get(y, 0.0)
            rev_oil[y] = oil_vol * oil_price
            rev_gas[y] = gas_vol * gas_price
            rev[y] = rev_oil[y] + rev_gas[y]
        self.annual_revenue = rev
        self.annual_revenue_oil = rev_oil
        self.annual_revenue_gas = rev_gas
        cumulative_revenue_array = np.cumsum([self.annual_revenue.get(y, 0.0) for y in self.all_years])
        for i, y in enumerate(self.all_years):
            self.annual_cum_revenue[y] = cumulative_revenue_array[i]
        return self.annual_revenue

    # ---------------------------
    # 세금
    # ---------------------------
    def _calculate_CIT(self, taxable_income: float)->float:
        taxable_income_krw = taxable_income * self.exchange_rate / 10
        if taxable_income_krw > 30000: CIT_krw = (6268 + (taxable_income_krw - 30000) * 0.24) * 1.1
        elif taxable_income_krw > 2000: CIT_krw = (378 + (taxable_income_krw - 2000) * 0.21) * 1.1
        elif taxable_income_krw > 20: CIT_krw = (1.8 + (taxable_income_krw - 2) * 0.19) * 1.1
        elif taxable_income_krw > 0: CIT_krw = (taxable_income_krw * 0.09) * 1.1
        else: return 0.0
        return CIT_krw / self.exchange_rate * 10

    def calculate_taxes(self, output=True):
        self._build_full_timeline()
        if not self.annual_revenue: self.calculate_annual_revenue(output=False)
        if not self.annual_depreciation: self.calculate_depreciation(output=False)
        for y in self.all_years:
            taxable = self.annual_revenue.get(y, 0.0) - (self.annual_opex.get(y, 0.0) + self.annual_abex.get(y, 0.0)) - self.annual_depreciation.get(y, 0.0)
            corp_tax = self._calculate_CIT(taxable)
            self.taxable_income[y] = taxable
            self.corporate_income_tax[y] = corp_tax
            self.annual_total_tax[y] = corp_tax
        self.total_tax = sum(self.annual_total_tax.values())

    def _calculate_royalty_rates(self, r_factor:float):
        if r_factor < 1.25: return 0.01
        elif r_factor < 3: return round(((18.28 * (r_factor-1.25)) + 1)/100,2)
        else: return 0.33

    def calculate_royalty(self):
        if not self.annual_revenue: self.calculate_annual_revenue(output=False)
        self._build_full_timeline()
        years = self.all_years
        annual_r_factor = {}
        annual_royalty = {}
        cum_royalty = 0
        for y in years:
            cum_rev_after = self.annual_cum_revenue.get(y, 0.0) - cum_royalty
            denom = self.annual_cum_capex_inflated.get(y, 0.0) + self.annual_cum_opex_inflated.get(y, 0.0)
            annual_r_factor[y] = cum_rev_after / denom if denom != 0 else 0.0
            rate = self._calculate_royalty_rates(annual_r_factor[y])
            annual_royalty[y] = self.annual_revenue.get(y, 0.0) * rate
            cum_royalty += annual_royalty[y]
        self.annual_r_factor = annual_r_factor
        self.annual_royalty = annual_royalty

    def calculate_net_cash_flow(self, cop=True, output=True, discovery_bonus: Optional[float] = None):
        self._build_full_timeline()
        if not self.annual_revenue: self.calculate_annual_revenue(output=False)
        if not self.annual_royalty: self.calculate_royalty()
        if not self.annual_total_tax: self.calculate_taxes(output=False)
        years = self.all_years
        cop_year = years[-1]
        if cop:
            for y in years:
                if y >= (self.production_start_year or 0):
                    if self.annual_revenue.get(y, 0.0) > 0 and self.annual_revenue.get(y, 0.0) < self.annual_opex_inflated.get(y, 0.0):
                        cop_year = y; break
        self.cop_year = cop_year
        self.annual_net_cash_flow = {}
        self.cumulative_cash_flow = {}
        cum_ncf = 0.0
        for y in years:
            rev = self.annual_revenue.get(y, 0.0) if y <= cop_year else 0.0
            royalty = self.annual_royalty.get(y, 0.0) if y <= cop_year else 0.0
            capex = self.annual_capex_inflated.get(y, 0.0)
            opex = self.annual_opex_inflated.get(y, 0.0) if y <= cop_year else 0.0
            abex = self.annual_abex_inflated.get(y, 0.0)
            tax = self.annual_total_tax.get(y, 0.0)
            other = self.other_fees.get(y, 0.0)
            bonus = (discovery_bonus if y == self.production_start_year else 0.0) if discovery_bonus else 0.0
            ncf = rev - royalty - (capex + opex + abex + bonus + other) - tax
            self.annual_net_cash_flow[y] = ncf
            cum_ncf += ncf
            self.cumulative_cash_flow[y] = cum_ncf
        return self.annual_net_cash_flow

    def calculate_npv(self, discount_rate: Optional[float] = None, output=True):
        if not self.annual_net_cash_flow: self.calculate_net_cash_flow(output=False)
        if discount_rate is not None: self.discount_rate = discount_rate
        years = np.array(self.all_years)
        dfs = 1.0 / ((1.0 + self.discount_rate) ** (years - years[0]))
        pv = {y: float(self.annual_net_cash_flow.get(y, 0.0)) * float(dfs[i]) for i, y in enumerate(years)}
        self.present_values = pv
        self.npv = sum(pv.values())
        return self.npv

    def calculate_irr(self):
        if not self.annual_net_cash_flow: return None
        cf_array = np.array([self.annual_net_cash_flow[y] for y in self.all_years])
        self.irr = round(npf.irr(cf_array), 4)
        return self.irr

    def get_project_summary(self):
        if self.npv is None: self.calculate_npv(output=False)
        self.calculate_irr()
        
        # Calculate payback year
        payback = None
        for y, val in self.cumulative_cash_flow.items():
            if y >= self.production_start_year and val >= 0:
                payback = y
                break
        self.payback_year = payback
        
        sum_val = lambda d: sum(d.values()) if d else 0.0
        return {
            'total_revenue': sum_val(self.annual_revenue),
            'total_royalty': sum_val(self.annual_royalty),
            'total_capex': sum_val(self.annual_capex_inflated),
            'total_opex': sum_val(self.annual_opex_inflated),
            'total_abex': sum_val(self.annual_abex_inflated),
            'total_tax': sum_val(self.annual_total_tax),
            'npv': self.npv, 'irr': self.irr,
            'payback_year': self.payback_year,
            'final_cumulative': self.cumulative_cash_flow.get(self.all_years[-1], 0.0) if self.all_years else 0.0
        }

    def to_df(self):
        cols = [self.annual_oil_production, self.oil_price_by_year, self.annual_gas_production, self.gas_price_by_year, self.annual_revenue, self.annual_royalty, self.annual_capex_inflated, self.annual_opex_inflated, self.annual_abex_inflated, self.other_fees, self.corporate_income_tax, self.annual_net_cash_flow]
        idx = ['석유 (MMbbl)', '유가 ($/bbl)', '가스 (BCF)', '가스가 ($/mcf)', '수익 (MM$)', '조광료 (MM$)', 'CAPEX (MM$)', 'OPEX (MM$)', 'ABEX (MM$)', 'Others (MM$)', '법인세 (MM$)', 'NCF (MM$)']
        df = pd.DataFrame(cols, index=idx)
        df.insert(0, 'Total', df.sum(axis=1))
        return df.dropna(axis=1, how='any')

# ---------------------------
# Multi-Company Configuration
# ---------------------------
class CompanyConfig(BaseModel):
    name: str
    pi: float  # Participating Interest (e.g., 0.51)
    farm_in_expo_share: Optional[float] = None  # e.g., 1.0 for 100% carry
    farm_in_expo_cap: Optional[float] = None    # MM$

class MultiCompanyCashFlow:
    def __init__(self, project_cf: CashFlowKOR, companies: List[CompanyConfig]):
        self.project_cf = project_cf
        self.companies = companies
        self.company_results: Dict[str, CashFlowKOR] = {}

    def calculate(self, output=False):
        if not self.project_cf.all_years: self.project_cf._build_full_timeline()
        if not self.project_cf.annual_revenue: self.project_cf.calculate_annual_revenue(output=False)
        if not self.project_cf.annual_royalty: self.project_cf.calculate_royalty()
        if not self.project_cf.annual_total_tax: self.project_cf.calculate_taxes(output=False)
        if not self.project_cf.annual_net_cash_flow: self.project_cf.calculate_net_cash_flow(output=False)
        project_years = self.project_cf.all_years
        if hasattr(self.project_cf.development_cost, 'get_cost_breakdown'):
            breakdown = self.project_cf.development_cost.get_cost_breakdown()
            project_capex_breakdown = breakdown.get('capex_breakdown', {})
            project_opex = breakdown.get('annual_opex', {})
            project_abex = breakdown.get('annual_abex', {})
        else:
            project_capex_breakdown = self.project_cf.capex_breakdown
            project_opex = self.project_cf.annual_opex
            project_abex = self.project_cf.annual_abex
        project_exploration = project_capex_breakdown.get('exploration', {})
        for comp in self.companies:
            ccf = CashFlowKOR(base_year=self.project_cf.base_year, oil_price_by_year=self.project_cf.oil_price_by_year, gas_price_by_year=self.project_cf.gas_price_by_year, cost_inflation_rate=self.project_cf.cost_inflation_rate, discount_rate=self.project_cf.discount_rate, exchange_rate=self.project_cf.exchange_rate)
            comp_capex_breakdown = {}; cum_proj_explo = 0.0; comp_exploration = {}
            for y in project_years:
                proj_explo_y = project_exploration.get(y, 0.0)
                if proj_explo_y == 0: comp_exploration[y] = 0.0; continue
                if comp.farm_in_expo_share is not None and comp.farm_in_expo_cap is not None:
                    cap = comp.farm_in_expo_cap; share = comp.farm_in_expo_share
                    if cum_proj_explo >= cap: comp_exploration[y] = proj_explo_y * comp.pi
                    else:
                        remaining_cap = cap - cum_proj_explo; portion_under_cap = min(proj_explo_y, remaining_cap); portion_over_cap = max(0, proj_explo_y - portion_under_cap)
                        comp_exploration[y] = (portion_under_cap * share) + (portion_over_cap * comp.pi)
                else: comp_exploration[y] = proj_explo_y * comp.pi
                cum_proj_explo += proj_explo_y
            comp_capex_breakdown['exploration'] = comp_exploration
            for key, annual_vals in project_capex_breakdown.items():
                if key == 'exploration': continue
                if isinstance(annual_vals, dict): comp_capex_breakdown[key] = {y: v * comp.pi for y, v in annual_vals.items()}
            comp_annual_capex = {y: sum(c.get(y,0) for c in comp_capex_breakdown.values()) for y in project_years}
            ccf.set_development_costs({'cost_years': project_years, 'annual_capex': comp_annual_capex, 'annual_opex': {y: v * comp.pi for y, v in project_opex.items()}, 'annual_abex': {y: v * comp.pi for y, v in project_abex.items()}, 'capex_breakdown': comp_capex_breakdown}, output=False)
            ccf.set_production_profile_from_dicts({y: v * comp.pi for y, v in self.project_cf.annual_oil_production.items()}, {y: v * comp.pi for y, v in self.project_cf.annual_gas_production.items()})
            ccf.other_fees = {y: v * comp.pi for y, v in self.project_cf.other_fees.items()}
            ccf.calculate_annual_revenue(output=False); ccf.annual_royalty = {y: v * comp.pi for y, v in self.project_cf.annual_royalty.items()}
            ccf.calculate_depreciation(method='straight_line', useful_life=10, output=False); ccf.calculate_taxes(output=False); ccf.calculate_net_cash_flow(output=False); ccf.calculate_npv(output=False); ccf.calculate_irr()
            self.company_results[comp.name] = ccf
        return self.company_results

    def get_summary_df(self) -> pd.DataFrame:
        data = []
        for name, ccf in self.company_results.items():
            summ = ccf.get_project_summary()
            data.append({'Company': name, 'PI (%)': [c.pi * 100 for c in self.companies if c.name == name][0], 'NPV (MM$)': summ['npv'], 'IRR (%)': summ['irr'] * 100 if isinstance(summ['irr'], (int, float)) else 0.0, 'Total Revenue (MM$)': summ['total_revenue'], 'Total CAPEX (MM$)': summ['total_capex'], 'Net Cash Flow (MM$)': summ['final_cumulative']})
        total_summ = self.project_cf.get_project_summary()
        data.append({'Company': 'PROJECT TOTAL', 'PI (%)': 100.0, 'NPV (MM$)': total_summ['npv'], 'IRR (%)': total_summ['irr'] * 100 if isinstance(total_summ['irr'], (int, float)) else 0.0, 'Total Revenue (MM$)': total_summ['total_revenue'], 'Total CAPEX (MM$)': total_summ['total_capex'], 'Net Cash Flow (MM$)': total_summ['final_cumulative']})
        return pd.DataFrame(data)

if __name__ == "__main__":
    pass