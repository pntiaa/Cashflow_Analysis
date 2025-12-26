# from pandas.core import base
import pandas as pd
from typing import Dict, Callable, Optional, Union, List
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

    annual_cum_revenue: Dict[str, Dict[int, float]] = Field(default_factory=dict)
    annual_cum_capex_inflated: Dict[str, Dict[int, float]] = Field(default_factory=dict)
    annual_cum_opex_inflated: Dict[str, Dict[int, float]] = Field(default_factory=dict)
    annual_cum_abex_inflated: Dict[str, Dict[int, float]] = Field(default_factory=dict)
    annual_r_factor: Dict[str, Dict[int, float]] = Field(default_factory=dict)
    annual_royalty: Dict[str, Dict[int, float]] = Field(default_factory=dict)

    # 생산량
    oil_production_series: Optional[Any] = None
    gas_production_series: Optional[Any] = None
    production_start_year: Optional[int] = None
    production_years: Optional[int] = None
    annual_oil_production: Dict[str, Dict[int, float]] = Field(default_factory=dict)
    annual_gas_production: Dict[str, Dict[int, float]] = Field(default_factory=dict)
    total_oil_production: float = None
    total_gas_production: float = None

    # 감가상각 (연도별 딕셔너리)
    annual_depreciation: Dict[int, float] = Field(default_factory=dict)

    # 연간 회계 관련 딕셔너리
    discovery_bonus: Optional[float]= None
    annual_discovery_bonus: Dict[int, float] = Field(default_factory=dict) # 연도별
    annual_revenue: Dict[int, float] = Field(default_factory=dict)
    annual_revenue_oil: Dict[int, float] = Field(default_factory=dict)
    annual_revenue_gas: Dict[int, float] = Field(default_factory=dict)
    annual_r_factor: Dict[int, float] = Field(default_factory=dict)
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
    payback: Optional[int] = None
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
        # 탐사 기간 포함 필요====================================================================================확인
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
        - 예상 속성/키: cost_years (리스트), annual_capex (딕셔너리), annual_opex (딕셔너리),
          annual_abex (딕셔너리), capex_breakdown (선택 사항).
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
        years = np.array(self.cost_years)
        years_from_start = years - years[0] # Convert to Python int for consistent key access
        inf = ((1.0 + self.cost_inflation_rate) ** years_from_start)
        for i, y in enumerate(years):
            self.annual_capex_inflated[y] = self.annual_capex.get(y, 0.0) * inf[i]
            self.annual_opex_inflated[y] = self.annual_opex.get(y, 0.0) * inf[i]
            self.annual_abex_inflated[y] = self.annual_abex.get(y, 0.0) * inf[i]

        cum_capex_inflated = np.cumsum([self.annual_capex_inflated.get(y, 0.0) for y in years])
        cum_opex_inflated = np.cumsum([self.annual_opex_inflated.get(y, 0.0) for y in years])
        cum_abex_inflated = np.cumsum([self.annual_abex_inflated.get(y, 0.0) for y in years])

        for i, y in enumerate(years):
            self.annual_cum_capex_inflated[y] =cum_capex_inflated[i]
            self.annual_cum_opex_inflated[y] =cum_opex_inflated[i]
            self.annual_cum_abex_inflated[y] =cum_abex_inflated[i]

        if output:
            print(f"[set_development_costs] 개발 연도: {self.cost_years}")
            print(f"  총 CAPEX (합계): {sum(self.annual_capex.values()):.3f} MM")
            print(f"  총 OPEX (합계): {sum(self.annual_opex.values()):.3f} MM")
            print(f"  총 ABEX (합계): {sum(self.annual_abex.values()):.3f} MM")
            print(f"  총 CAPEX Inflated (합계): {sum(self.annual_capex_inflated.values()):.3f} MM")
            print(f"  총 OPEX Inflated (합계): {sum(self.annual_opex_inflated.values()):.3f} MM")
            print(f"  총 ABEX Inflated (합계): {sum(self.annual_abex_inflated.values()):.3f} MM")
            # print(f"  CAPEX : {self.annual_capex}")
            # print(f"  CAPEX Inflated : {self.annual_capex_inflated}")
    # ---------------------------
    # 생산량 설정 함수
    # ---------------------------
    def set_production_profile_from_arrays(self, oil_prod, gas_prod, production_start_year: int):
        """
        - 석유 및 가스 (연간 물량) 배열/리스트와 생산 시작 연도 수용.
        - {연도: 물량} 딕셔너리로 변환.
        """
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
        """
        - {연도: 물량} 딕셔너리 수용.
        - `production_start_year`와 `production_years` 설정.
        """
        self.annual_oil_production = {int(k): float(v) for k, v in oil_dict.items()}
        self.annual_gas_production = {int(k): float(v) for k, v in gas_dict.items()}
        years = sorted(set(self.annual_oil_production.keys()) | set(self.annual_gas_production.keys()))
        if len(years) == 0:
            raise ValueError("생산량 딕셔너리가 비어 있습니다")
        self.production_start_year = years[0]
        self.production_years = len(years)

        self.total_gas_production = self._sum_dict_values(self.annual_gas_production)
        self.total_oil_production = self._sum_dict_values(self.annual_oil_production)

    # ---------------------------
    # 감가상각
    # ---------------------------
    def calculate_depreciation(self, method: str = 'production_base',
                               useful_life: int = 10,
                               depreciable_components: Optional[List[str]] = None,
                               output=True):
        """
        - 전체 프로젝트 기간에 걸쳐 연간 감가상각 딕셔너리 생성.
        - `depreciable_components`: `dev_capex_breakdown` 딕셔너리에 존재하는 키 목록 (예: ['drilling','subsea','FPSO',...])
        - None인 경우, 사용 가능한 drilling+subsea+FPSO+export_pipeline+terminal+PM_others을 합산.
        """
        if self.development_cost is None:
            raise ValueError("개발 비용을 먼저 설정하십시오")

        # 감가상각 총액 (MM$ 단위)
        if depreciable_components is None:
            # 세부 내역이 존재하는 경우 일반적인 키 합산 시도
            try:
                breakdown = self.capex_breakdown
                keys = ['drilling', 'subsea', 'FPSO', 'export_pipeline', 'terminal', 'PM_others']
                total_depr = 0.0
                for k in keys:
                    # breakdown[k]는 연도-값 딕셔너리일 수 있음; 딕셔너리인 경우 합산
                    val = breakdown.get(k)
                    if isinstance(val, dict):
                        total_depr += sum(val.values())
                    elif isinstance(val, (int, float)):
                        total_depr += float(val)
            except Exception:
                total_depr = sum(self.annual_capex.values())
        else:
            total_depr = 0.0
            for k in depreciable_components:
                val = self.dev_capex_breakdown.get(k, {})
                if isinstance(val, dict):
                    total_depr += sum(val.values())
                elif isinstance(val, (int, float)):
                    total_depr += float(val)

        # 전체 타임라인 연도 먼저 구축
        self._build_full_timeline()
        years = self.all_years

        # 연도별 초기화
        self.annual_depreciation = {y: 0.0 for y in years}

        # 매장량 설정
        if self.total_gas_production is None:
            self.total_gas_production = sum(self.annual_gas_production.values()) if self.annual_gas_production else 0.0

        remaining_reserve = self.total_gas_production

        if method == 'production_base':
            total_depr_amount= 0.0  # Renamed to avoid conflict with initial 'total_depr'
            # cum_capex_values = [self.annual_cum_capex_inflated.get(y, 0.0) for y in years] # Pre-calculate cumulative CAPEX values

            for y in years:
                gas_prod = self.annual_gas_production.get(y, 0.0)
                if remaining_reserve > 0 and gas_prod > 0:
                    # 감가상각 = (금년 누적CAPEX  - 이전 해 까지 누적 감가상각) *  (금년 생산량) / (잔여 매장량)
                    current_cum_capex = self.annual_cum_capex_inflated.get(y, 0.0)
                    if remaining_reserve > 0:
                        ratio = gas_prod / remaining_reserve
                        dep = (current_cum_capex - total_depr_amount) * ratio
                        self.annual_depreciation[y] = dep
                        total_depr_amount += dep
                        remaining_reserve -= gas_prod
                else:
                    self.annual_depreciation[y] = 0.0 # No production or remaining reserve, no depreciation

        elif method == 'straight_line':
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
            raise ValueError("알 수 없는 감가상각 방법입니다")

        if output:
            print(f"[depr] 방법={method}, 총 감가상각={total_depr_amount:.3f} MM") # Changed to total_depr_amount for accurate output
            return self.annual_depreciation

    # ---------------------------
    # 수익 계산
    # ---------------------------
    def calculate_annual_revenue(self, output=True):
        """
        - 연간 수익 딕셔너리 {연도: MM$} 계산.
        - 수익 = 석유 생산량(MMbbl) * 석유 가격($/bbl) + 가스 생산량(bcf) * 가스 가격($/mcf).
        """
        if not self.annual_oil_production and not self.annual_gas_production:
            raise ValueError("생산 프로필이 설정되지 않았습니다")

        # 필수: 전체 프로젝트 연도 생성
        self._build_full_timeline()

        # Use self.all_years for revenue calculations to cover the full timeline
        # Initialize revenue dicts for all project years
        rev = {y: 0.0 for y in self.all_years}
        rev_oil = {y: 0.0 for y in self.all_years}
        rev_gas = {y: 0.0 for y in self.all_years}

        # Calculate revenue only for years with production data
        production_years_union = sorted(set(self.annual_oil_production.keys()) | set(self.annual_gas_production.keys()))
        for y in production_years_union:
            oil_vol = self.annual_oil_production.get(y, 0.0)
            gas_vol = self.annual_gas_production.get(y, 0.0)
            oil_price = self.oil_price_by_year.get(y, 0.0)
            gas_price = self.gas_price_by_year.get(y, 0.0)
            revenue_oil = (oil_vol * oil_price)
            revenue_gas = (gas_vol * gas_price)
            revenue = revenue_oil + revenue_gas
            rev_oil[y] = float(revenue_oil)
            rev_gas[y] = float(revenue_gas)
            rev[y] = float(revenue)

        # Assign the calculated annual revenues to the class attributes
        self.annual_revenue = rev
        self.annual_revenue_oil = rev_oil
        self.annual_revenue_gas = rev_gas

        # Calculate cumulative revenue for all_years
        cumulative_revenue_array = np.cumsum([self.annual_revenue.get(y, 0.0) for y in self.all_years])
        for i, y in enumerate(self.all_years):
            self.annual_cum_revenue[y] = cumulative_revenue_array[i]

        if output:
            print(f"[revenue] 총 수익 (MM$): {sum(self.annual_revenue.values()):.3f}")
            return self.annual_revenue

    # ---------------------------
    # 세금
    # ---------------------------
    def _calculate_CIT(self, taxable_income: float)->float:
        '''
        법인세율 계산 함수.
        - MM$ 단위를 천만원으로 변환하여 계산 후, 다시 MM$로 변환.
        '''
        taxable_income_krw = taxable_income * self.exchange_rate / 10
        if taxable_income_krw > 30_000:
            CIT_krw = (6268 + ( taxable_income_krw - 30_000) * 0.24) * 1.1
            CIT = CIT_krw / self.exchange_rate * 10 # 백만 USD로 변환
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
        - 연간 과세 소득, 법인세, 지방세, 총 세금 계산.
        - 과세 소득 = 수익 - (운영 비용 + 폐쇄 비용) - 감가상각.
        - 과세 소득이 0보다 큰 경우에만 세금 계산.
        """
        # 전체 타임라인 구축
        self._build_full_timeline()

        # 수익 존재 여부 확인
        if not self.annual_revenue:
            self.calculate_annual_revenue()

        # 감가상각 존재 여부 확인
        if not self.annual_depreciation:
            self.calculate_depreciation()

        self.taxable_income = {}
        self.corporate_income_tax = {}
        self.annual_total_tax = {}
        self.total_tax:float = None

        for y in self.all_years:
            rev = float(self.annual_revenue.get(y, 0.0))
            opex = float(self.annual_opex.get(y, 0.0))
            abex = float(self.annual_abex.get(y, 0.0))
            depr = float(self.annual_depreciation.get(y, 0.0))
            taxable = rev - (opex + abex) - depr
            # 양수 과세 소득에 대해서만 과세
            corp_tax = self._calculate_CIT(taxable)
            total = corp_tax
            self.taxable_income[y] = taxable
            self.corporate_income_tax[y] = corp_tax
            self.annual_total_tax[y] = total

        self.total_tax = sum(self.annual_total_tax.values())
        if output:
            print(f"[taxes] 총 세금 (MM$): {sum(self.annual_total_tax.values()):.3f}"
            f"(KRW: {sum(self.annual_total_tax.values()) * self.exchange_rate:.0f})")
            return self.annual_total_tax,

    def _calculate_royalty_rates(self, r_factor:float):
        # R-Factor에 따른 조광료율 계산
        if r_factor < 1.25:
            royalty_rates = 0.01
        elif r_factor <3:
            royalty_rates = round(((18.28 * (r_factor-1.25)) + 1)/100,2)
        else:
            royalty_rates = 0.33
        return royalty_rates

    def calculate_royalty(self):
        # 수익 계산 여부 확인
        if not self.annual_revenue:
            self.calculate_annual_revenue()
        # 개발비용 설정 확인
        if not self.annual_capex:
            raise ValueError("개발비용이 설정되지 않았습니다")

        # 전체 타임라인 구축
        self._build_full_timeline()
        years = list(self.all_years)

        annual_r_factor = {}
        annual_royalty_rates = {}
        annual_royalty = {}
        cum_revenue_after_royalty = {}
        cum_royalty = 0

        for y in years:
            # r-factor 산식 : (누적 매출액 - 누적 로열티) / 누적 (CAPEX+누적 OPEX)
            cum_revenue_after_royalty[y] = self.annual_cum_revenue.get(y, 0.0) - cum_royalty
            # Ensure denominators are not zero before division
            denominator = self.annual_cum_capex_inflated.get(y, 0.0) + self.annual_cum_opex_inflated.get(y, 0.0)
            if denominator != 0:
                annual_r_factor[y] = cum_revenue_after_royalty[y] / denominator
            else:
                annual_r_factor[y] = 0.0 # Or handle as appropriate, e.g., np.inf if cum_revenue_after_royalty[y] > 0

            annual_royalty_rates[y] = self._calculate_royalty_rates(annual_r_factor[y])
            # '조광료'는 부과대상연도의 매출액에 조광료 부과요율을 곱하여 산정한다.
            annual_royalty[y] = self.annual_revenue.get(y, 0.0) * annual_royalty_rates[y] # Use .get for safety
            cum_royalty +=  annual_royalty[y]

        self.annual_r_factor = annual_r_factor
        self.annual_royalty_rates = annual_royalty_rates
        self.cum_revenue_after_royalty = cum_revenue_after_royalty
        self.annual_royalty = annual_royalty

        return None

    def calculate_public_water_fee(self,
                                   occupying_area=0.15,
                                   base_land_price = 271_855):
        '''
        공유수면 사=용 점용료 [공유수면 관리 및 매립에 관한 법률]
         - occupying_area (km2)
         - base_land_price (원)
          * 점용료 사용료 = 공유수면 사용면적(m2) x 인접한 토지가격(공시지가) x 15%
          * 2년 이상 점용 사용한 경우 연간  점용료 사용료가 전년보다 10%이상 증가한 경우, 다음의 금액을 추가 납부 (증가율 조정)

         <기본계산값>
         # 생산시설 면적 : 7,500 m2
         # 해저배관 : 142,500 m2
         # 생산시설 공시지가 : 43,000원/m2 (울산 동구 일산동 905-7)
         # 해저배관 공시지가 : 283,900원/m2 (울산 울주군 온산읍 당월리 505)
         # = (7,500 *  43,000) + (142,500 *  283,900)
        '''
        land_price_usd = (base_land_price / self.exchange_rate) # $/m2
        public_water_fee = occupying_area * land_price_usd * 0.15 # 15% in MM$

        # 전체 타임라인 구축
        self._build_full_timeline()
        years = np.array(self.all_years)
        years_from_start = years - years[0]
        inf = ((1.0 + self.cost_inflation_rate) ** years_from_start)

        annual_public_water_fee = {}
        for i, y in enumerate(years):
            annual_public_water_fee[y] = public_water_fee * inf[i]
            self.other_fees[y] = self.other_fees.get(y, 0.0) + annual_public_water_fee[y]
            # self.annual_opex_inflated[y] = self.annual_opex_inflated.get(y, 0.0) + annual_public_water_fee[y]
        print(f"공유수면 점사용료 : 총 {sum(annual_public_water_fee.values())}MM$")
        return self.other_fees

    def calculate_education_fund(self):
        '''
        조광계약상 정의된 교육훈련비
        '''
        # 전체 타임라인 구축
        self._build_full_timeline()
        years = self.all_years
        education_fund = 25.0
        annual_education_fund = {}
        for y in years:
            annual_education_fund[y] = education_fund
            self.other_fees[y] = self.other_fees.get(y, 0.0) + annual_education_fund[y]
        print(f"교육훈련비 : 총 {sum(annual_education_fund.values())}MM$")
        return self.other_fees

    # ---------------------------
    # 순현금흐름 (세후)
    # ---------------------------
    def calculate_net_cash_flow(self,
                                cop =True,
                                output = True,
                                discovery_bonus: Optional[float] = None,):
        """
        - 수익, CAPEX, OPEX, ABEX, 세금을 결합하여 연간 순현금흐름 및 누적 계산.
        - 공식 (세후): 순현금흐름 = 수익 - (CAPEX + OPEX + ABEX) - 총 세금.
        - 모든 값은 연간 MM$ 단위.
        """
        if self.development_cost is None:
            raise ValueError("개발 비용이 설정되지 않았습니다")

        # 1. 사전 계산 확인
        self._build_full_timeline()
        if not self.annual_revenue: self.calculate_annual_revenue(output=False)
        if not self.annual_royalty: self.calculate_royalty()
        if not self.annual_total_tax: self.calculate_taxes(output=False)

        years = self.all_years

        # 2. COP(생산중단) 시점 판별 및 ABEX 시점 추출
        # 원래 계획된 ABEX가 발생하는 첫 해를 찾음
        original_abex_year = None
        sorted_abex_years = sorted([y for y, v in self.annual_abex_inflated.items() if v > 0])
        if sorted_abex_years:
            original_abex_year = sorted_abex_years[0]
            total_abex_value = sum(self.annual_abex_inflated.values())
        else:
            total_abex_value = 0.0

        cop_year = years[-1]  # 기본값은 프로젝트 마지막 해
        # ---  COP(Economic Limit) 판정  ---
        if cop:
            for y in years:
                # 생산 시작 이후부터 체크 (수익이 발생하기 시작하는 시점부터)
                if y >= self.production_start_year:
                    rev = self.annual_revenue.get(y, 0.0)
                    opex = self.annual_opex_inflated.get(y, 0.0)
                    # 경제성 한계 판별: 매출 < 운영비 (수익이 0보다 큰 경우에만 체크)
                    if rev > 0 and rev < opex:
                        cop_year = y
                        # print(cop_year)
                        break
            self.cop_year = cop_year
            # --- COP 기준 프로필 필터링 (Re-profiling) ---
            # COP 이후 수익과 운영비를 제거한 "Actual" 딕셔너리로 교체
            actual_revenue = {y: (self.annual_revenue.get(y, 0.0) if y <= cop_year else 0.0) for y in years}
            actual_opex = {y: (self.annual_opex_inflated.get(y, 0.0) if y <= cop_year else 0.0) for y in years}

            # --- ABEX 재배치 --------------
            actual_abex = {y: 0.0 for y in years}
            orig_abex_sch = {y: v for y, v in self.annual_abex_inflated.items() if v > 0}
            if orig_abex_sch:
                sorted_orig = sorted(orig_abex_sch.keys())
                first_orig = sorted_orig[0]
                for y_orig, val in orig_abex_sch.items():
                    new_y = cop_year + (y_orig - first_orig)
                    if new_y in actual_abex: actual_abex[new_y] += val
                    else: actual_abex[years[-1]] += val

            # royalty 재계산을 위해 속성 업데이트
            self.annual_revenue = actual_revenue
            self.annual_opex_inflated = actual_opex
            self.annual_abex_inflated = actual_abex

            # 누적 수익(Cumulative Revenue) 재계산 (R-Factor 영향을 위함)
            cum_rev = 0.0
            for y in years:
                cum_rev += self.annual_revenue[y]
                self.annual_cum_revenue[y] = cum_rev

            # --- 조광료(Royalty) 재계산 ---
            # 줄어든 누적 수익과 조정된 운영비를 바탕으로 R-Factor 및 조광료 다시 계산
            self.calculate_royalty()

            # ---  세금(CIT) 재계산 ---
            self.calculate_taxes(output=False)


        # 4. 최종 NCF 계산 (조정된 값들 적용)
        self.annual_net_cash_flow = {}
        self.cumulative_cash_flow = {}
        cum_ncf = 0.0

        # 발견보너스가 있으면, 삽입
        if discovery_bonus:
            self.discovery_bonus = discovery_bonus
            bonus={}
            y =  self.production_start_year
            bonus[y] = discovery_bonus
            for y in years:
                self.annual_discovery_bonus[y] = bonus.get(y, 0.0)

        for y in years:
            rev = self.annual_revenue.get(y, 0.0)
            royalty = self.annual_royalty.get(y, 0.0)
            capex = self.annual_capex_inflated.get(y, 0.0)
            opex = self.annual_opex_inflated.get(y, 0.0)
            abex = self.annual_abex_inflated.get(y, 0.0) # 조정된 ABEX 적용
            tax = self.annual_total_tax.get(y, 0.0)
            other = self.other_fees.get(y, 0.0)
            bonus = self.annual_discovery_bonus.get(y, 0.0) if self.discovery_bonus else 0.0

            ncf = rev - royalty - (capex + opex + abex + bonus + other) - tax
            self.annual_net_cash_flow[y] = ncf
            cum_ncf += ncf
            self.cumulative_cash_flow[y] = cum_ncf

        if output:
            if cop:
                print(f"[COP 적용] 생산 종료: {cop_year}년")
            print(f"총 수익: {sum(self.annual_revenue.values()):.2f} MM$")
            print(f"총 CAPEX: {sum(self.annual_capex_inflated.values()):.2f} MM$")
            print(f"총 OPEX: {sum(self.annual_opex_inflated.values()):.2f} MM$")
            print(f"총 ABEX: {sum(self.annual_abex_inflated.values()):.2f} MM$")
            print(f"총 Others: {sum(self.other_fees.values()):.2f} MM$")
            print(f"총 조광료: {sum(self.annual_royalty.values()):.2f} MM$")
            print(f"총 세금: {sum(self.annual_total_tax.values()):.2f} MM$")
            print(f"최종 NCF: {cum_ncf:.2f} MM$")

        return self.annual_net_cash_flow

    # ---------------------------
    # NPV 및 IRR
    # ---------------------------
    def calculate_npv(self, discount_rate: Optional[float] = None, output=True):
        # 연간 순현금흐름이 없으면 계산
        if self.annual_net_cash_flow is None or len(self.annual_net_cash_flow) == 0:
            self.calculate_net_cash_flow()
        # 할인율이 제공되면 업데이트
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
            print(f"[npv] 할인율={self.discount_rate:.3f}, NPV={self.npv:.3f} MM")
            return self.npv

    def calculate_irr(self, max_iter: int = 200, tol: float = 1e-6):
        """
        numpy_financial 패키지에서 제공하는 함수로 대체
        예시) 입력 : round(npf.irr([-100, 39, 59, 55, 20]), 5) 출력 : 0.28095
        - 입력 : valuesarray_like, shape(N,)
        - 출력 : float / Internal Rate of Return for periodic input values.
        """
        if self.annual_net_cash_flow is None or len(self.annual_net_cash_flow) == 0:
            return None

        # 배열 구축
        cf_array = np.array([self.annual_net_cash_flow[y] for y in self.all_years])

        # IRR 계산
        irr  = round(npf.irr(cf_array), 4) # 퍼센트로 표기시 소숫점 둘째짜리까지 표시되도록
        self.irr = irr
        return irr

    # ---------------------------
    # 요약
    # ---------------------------
    def get_project_summary(self):
        """
        - 주요 지표 딕셔너리 반환 (NPV, IRR, 회수 기간, 총계).
        """
        if self.npv is None:
            self.calculate_npv()

        self.irr = self.calculate_irr()
        # 총 자원량
        self.total_oil_production = sum(self.annual_oil_production.values()) if self.annual_oil_production else 0.0
        self.total_gas_production = sum(self.annual_gas_production.values()) if self.annual_gas_production else 0.0
        self.total_revenue = sum(self.annual_revenue.values()) if self.annual_revenue else 0.0
        self.total_royalty = sum(self.annual_royalty.values()) if self.annual_royalty else 0.0
        self.total_capex = sum(self.annual_capex_inflated.values()) if self.annual_capex else 0.0
        self.total_opex = sum(self.annual_opex_inflated.values()) if self.annual_opex else 0.0
        self.total_abex = sum(self.annual_abex_inflated.values()) if self.annual_abex else 0.0
        self.total_tax = sum(self.annual_total_tax.values()) if self.annual_total_tax else 0.0

        # 회수 연도
        payback = None # 회수 연도 초기화
        for y, val in self.cumulative_cash_flow.items():
            if val >= 0:
                payback = y
                break
        self.payback = payback

        return {
            'total_oil_production': self.total_oil_production,
            'total_gas_production': self.total_gas_production,
            'total_revenue': self.total_revenue,
            'total_royalty': self.total_royalty,
            'total_capex': self.total_capex,
            'total_opex': self.total_opex,
            'total_abex': self.total_abex,
            'total_tax': self.total_tax,
            'npv': self.npv,
            'irr': self.irr,
            'payback_year': self.payback,
           'final_cumulative': self.cumulative_cash_flow.get(self.all_years[-1], 0.0)
        }
    def to_df(self):
        # 3. 생산량 및 유가 확인
        cols = [
            self.annual_oil_production,
            self.oil_price_by_year,
            self.annual_gas_production,
            self.gas_price_by_year,
            self.annual_revenue_oil,
            self.annual_revenue_gas,
            self.annual_revenue,
            self.annual_royalty,
            self.annual_discovery_bonus,
            self.annual_capex_inflated,
            self.annual_opex_inflated,
            self.annual_abex_inflated,
            self.other_fees,
            self.corporate_income_tax,
            self.annual_net_cash_flow
            ]
        idx = [
            '석유 - MMbbls',
            '유가 - $/bbls',
            '가스 - BCF',
            '가스 - $/mcf',
            '석유 판매 - MM$',
            '가스 판매 - MM$',
            '수익 - MM$',
            '조광료 - MM$',
            '발견보너스 - MM$',
            'CAPEX - MM$',
            'OPEX - MM$',
            'ABEX - MM$',
            'Others - MM$',
            '법인세 - MM$',
            'NCF - MM$'
            ]

        df = pd.DataFrame(cols, index=idx)
        # 제일 앞 행에 합계 삽입
        df.insert(0, 'Total', df.sum(axis=1))
        #  제일 마지막 행에 삽입할 경우
        # df['Total'] = df.sum(axis=1)
        df_cleaned = df.dropna(axis=1, how='any')
        return df_cleaned

if __name__ == "__main__":
    pass    