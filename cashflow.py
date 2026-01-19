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
    capex_breakdown_inflated: Dict[str, Dict[int, float]] = Field(default_factory=dict)

    annual_cum_revenue: Dict[int, float] = Field(default_factory=dict)
    annual_cum_capex_inflated: Dict[int, float] = Field(default_factory=dict)
    annual_cum_opex_inflated: Dict[int, float] = Field(default_factory=dict)
    annual_cum_abex_inflated: Dict[int, float] = Field(default_factory=dict)
    annual_r_factor: Dict[int, float] = Field(default_factory=dict)
    annual_royalty: Dict[int, float] = Field(default_factory=dict)
    annual_high_price_royalty: Dict[int, float] = Field(default_factory=dict)

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
    loss_carryforward: Dict[int, float] = Field(default_factory=dict)
    corporate_income_tax: Dict[int, float] = Field(default_factory=dict)
    other_fees: Dict[int, float] = Field(default_factory=dict)  # 공유수면점사용료, 교육훈련비
    annual_total_tax: Dict[int, float] = Field(default_factory=dict)
    annual_investment_tax_credit: Dict[int, float] = Field(default_factory=dict)
    tax_credits_carried_forward: Dict[int, float] = Field(default_factory=dict)
    utilized_tax_credit: Dict[int, float] = Field(default_factory=dict)
    annual_rural_development_tax: Dict[int, float] = Field(default_factory=dict)
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
            # breakdown inflation 초기화
            self.capex_breakdown_inflated = {k: {} for k in self.capex_breakdown.keys()}

            for i, y in enumerate(years):
                self.annual_capex_inflated[y] = self.annual_capex.get(y, 0.0) * inf[i]
                self.annual_opex_inflated[y] = self.annual_opex.get(y, 0.0) * inf[i]
                self.annual_abex_inflated[y] = self.annual_abex.get(y, 0.0) * inf[i]
                
                # 각 breakdown 요소에 inflation 적용
                for k, component_dict in self.capex_breakdown.items():
                    if isinstance(component_dict, dict):
                        self.capex_breakdown_inflated[k][y] = component_dict.get(y, 0.0) * inf[i]

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
    def calculate_depreciation(self, method: str = 'unit_of_production', useful_life: int = 10, depreciable_components: Optional[List[str]] = None, output=True):
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

        if method == 'unit_of_production':
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
    # 기타 비용
    # ---------------------------
    def calculate_annual_other_cost(self):
        '''
        공유수면 점사용료(public water occupacy or usage fee) : 매년 정해진 금액을 지출, 100만불
        교육훈련비(training cost) : 매년 정해진 금액을 지출, 25만불
        '''
        self._build_full_timeline()
        years = self.all_years
        self.annual_other_cost = {y: 0.0 for y in years}

        for y in years:
            self.annual_other_cost[y] = 100 + 25
        return self.annual_other_cost

    # ---------------------------
    # 세금
    # ---------------------------   
    def _calculate_CIT(self, taxable_income: float)->float:
        '''
        Corporate Income Tax (CIT) 계산
        백만불(MM$) / 환율(MM$/KRW) => 백만원 / 10 => 천만원 단위로 계산
        지방세 = 법인세 * 10% 는 Total Tax 계산시에 반영
        '''
        taxable_income_krw = taxable_income * self.exchange_rate / 10
        if taxable_income_krw > 30000: CIT_krw = (6268 + (taxable_income_krw - 30000) * 0.24)
        elif taxable_income_krw > 2000: CIT_krw = (378 + (taxable_income_krw - 2000) * 0.21)
        elif taxable_income_krw > 20: CIT_krw = (1.8 + (taxable_income_krw - 2) * 0.19)
        elif taxable_income_krw > 0: CIT_krw = (taxable_income_krw * 0.09)
        else: return 0.0

        return CIT_krw / self.exchange_rate * 10

    def _calculate_royalty_rates(self, r_factor:float):
        if r_factor < 1.25: return 0.01
        elif r_factor < 3: return round(((18.28 * (r_factor-1.25)) + 1)/100,2)
        else: return 0.33

    def _calculate_investment_tax_credit(self):
        '''
        조세특례제한법 제24조 - 통합투자세액공제(Integrated Investment Tax Credit)
        Investment Tax Credit (ITC) Carryforward
        기본공제금액(A) : 해당 과세연도에 투자한 금액에 다음의 구분에 따른 비율을 곱한 금액에 상당하는 금액. 해당 프로젝트는 100분의 1
        추가공제금액(B) : 해당 과세연도에 투자한 금액이 해당 과세연도의 직전 3년간 연평균 투자 또는 취득금액을 초과하는 경우에는 그 초과하는 금액의 100분의 10에 상당하는 금액
        통합투자세액공제 대상자산 : 감가상각대상자산 - (Pipeline CAPEX + Pipeline Acquisition Tax + Pre-sanction Cost + PM & Other CAPEX)
        다만, 추가공제 금액이 기본공제 금액을 초과하는 경우에는 기본공제 금액의 2배를 한도로 함.
'
        조세특례제한법 제144조 제1항
        조세특례제한법상의 특정 세액공제(제24조 통합투자세액공제 포함)를 적용받을 때, 
        결손이 발생하여 납부할 세액이 없거나 최저한세 규정에 걸려 해당 연도에 공제받지 못한 금액이 있는 경우, 
        이를 다음 과세연도 개시일부터 10년 이내에 종료하는 각 과세연도로 이월하여 공제할 수 있습니다.
        '''
        self._build_full_timeline()
        years = self.all_years
        tax_credit = {y: 0.0 for y in years}
        
        # 대상 자산 산출을 위한 breakdown 확인
        breakdown = self.capex_breakdown_inflated
        # 대상 자산 = drilling + subsea + FPSO + terminal
        # (Pipeline CAPEX, Study costs, PM/Others 제외)
        target_keys = ['drilling', 'subsea', 'FPSO', 'terminal']
        
        target_investment = {y: 0.0 for y in years}
        for y in years:
            inv_y = 0.0
            for k in target_keys:
                val = breakdown.get(k, {})
                if isinstance(val, dict):
                    inv_y += val.get(y, 0.0)
                elif isinstance(val, (int, float)) and y == self.cost_years[0]: # dict가 아니면 첫해에 몰려있다고 가정 (보통 dict임)
                    inv_y += float(val)
            target_investment[y] = inv_y

        for i, y in enumerate(years):
            # 기본공제 (A): 1%
            A = target_investment[y] * 0.01
            
            # 추가공제 (B): 직전 3개년 평균 초과분의 10%
            # i=0: 직전 3개년 없음 (0)
            # i=1: 직전 1개년
            # i=2: 직전 2개년
            # i>=3: 직전 3개년
            prev_investments = []
            for j in range(max(0, i-3), i):
                prev_investments.append(target_investment[years[j]])
            
            if prev_investments:
                avg_prev = sum(prev_investments) / len(prev_investments)
                B = max(0, (target_investment[y] - avg_prev) * 0.10)
            else:
                B = 0.0
                
            tax_credit[y] = A + min(2*A, B)
            
        self.annual_investment_tax_credit = tax_credit
        return tax_credit

    def calculate_high_price_royalty(self, output=False):
        '''
        High Price Royalty(고유가 추가 조광료):
        아래의 조건이 모두 충족할 경우에 부과
        1. 원유 또는 천연가스의 평균 판매가격이 직전 5개년도 평균 판매가격 대비 20% 이상 높고,
        2. 원유가격이 배럴당 85불 이상일 경우
        [추가 조광료 = (부과대상연도의 판매가격 - 직전 5년 평균 판매가격 * 1.2) * 판매량 * 33%]
        '''
        if not self.annual_revenue:
            raise ValueError("수익을 먼저 계산해야 합니다")
        self._build_full_timeline()

        for y in self.all_years:
            oil_price_y = self.oil_price_by_year.get(y, 0.0)
            gas_price_y = self.gas_price_by_year.get(y, 0.0)

            # 2번 조건: 원유가격이 배럴당 85불 이상
            if oil_price_y < 85.0:
                continue

            # 직전 5개년도 평균 가격 계산
            prev_5_years = range(y - 5, y)
            oil_prices_prev = [self.oil_price_by_year.get(y, 0.0) for y in prev_5_years]
            gas_prices_prev = [self.gas_price_by_year.get(y, 0.0) for y in prev_5_years]

            avg_oil_prev = sum(oil_prices_prev) / 5 if oil_prices_prev else 0.0
            avg_gas_prev = sum(gas_prices_prev) / 5 if gas_prices_prev else 0.0

            additional_royalty = 0.0

            # 원유 추가 조광료 계산
            if avg_oil_prev > 0 and oil_price_y >= (avg_oil_prev * 1.2):
                oil_vol = self.annual_oil_production.get(y, 0.0)
                additional_royalty += (oil_price_y - avg_oil_prev * 1.2) * oil_vol * 0.33

            # 천연가스 추가 조광료 계산
            if avg_gas_prev > 0 and gas_price_y >= (avg_gas_prev * 1.2):
                gas_vol = self.annual_gas_production.get(y, 0.0)
                additional_royalty += (gas_price_y - avg_gas_prev * 1.2) * gas_vol * 0.33
            self.annual_high_price_royalty[y] = additional_royalty
        
        return self.annual_high_price_royalty
        
    def determine_cop_year(self, output=False):
        """
        COP 연도 결정: Revenue < (OPEX + Royalty) 인 첫 번째 연도
        """
        self._build_full_timeline()
        if not self.annual_revenue:
            self.calculate_annual_revenue(output=False)
        
        cop_year = self.all_years[-1]  # 기본값: 마지막 연도
        
        # Royalty 계산을 위한 누적 변수
        cum_royalty_temp = 0.0
        
        for y in self.all_years:
            if y >= (self.production_start_year or 0):
                revenue_y = self.annual_revenue.get(y, 0.0)
                opex_y = self.annual_opex_inflated.get(y, 0.0)
                
                # Royalty 실시간 계산 (R-factor 기반)
                # Note: COP 결정용이므로 COP 적용 전의 누적 비용/수익 사용
                denom = (self.annual_cum_capex_inflated.get(y, 0.0) 
                        + self.annual_cum_opex_inflated.get(y, 0.0) 
                        + self.annual_cum_abex_inflated.get(y, 0.0))
                
                cum_rev_after = self.annual_cum_revenue.get(y, 0.0) - cum_royalty_temp
                r_factor_y = cum_rev_after / denom if denom != 0 else 0.0
                rate_y = self._calculate_royalty_rates(r_factor_y)
                royalty_y = revenue_y * rate_y
                
                # OPEX가 존재하고, 수익이 (OPEX + Royalty)보다 작으면 생산 중단
                if opex_y > 0 and revenue_y < (opex_y + royalty_y):
                    cop_year = y
                    if output:
                        print(f"[COP] Year {y}: Revenue ({revenue_y:.2f}) < OPEX ({opex_y:.2f}) + Royalty ({royalty_y:.2f})")
                    break
                
                # 중단되지 않았다면 누적 Royalty 업데이트
                cum_royalty_temp += royalty_y
        
        self.cop_year = cop_year
        
        if output:
            print(f"[COP] Determined COP Year: {cop_year}")
        
        return cop_year
    
    def apply_cop_adjustments(self, output=False):
        """
        COP 이후의 모든 생산/비용/수익 조정을 한 번에 처리
        이 함수 호출 후에는 조광료, 세금 등을 재계산해야 함
        """
        if self.cop_year is None:
            raise ValueError("COP 연도를 먼저 결정해야 합니다 (determine_cop_year 호출)")
        
        cop_year = self.cop_year
        years = self.all_years
        
        if output:
            print(f"\n{'='*60}")
            print(f"[COP Adjustments] Applying changes for COP Year: {cop_year}")
            print(f"{'='*60}")
        
        # ===== 1. 생산량 조정 (COP 이후 0) =====
        original_gas_production = sum(self.annual_gas_production.values())
        original_oil_production = sum(self.annual_oil_production.values())
        
        for y in years:
            if y > cop_year:
                self.annual_gas_production[y] = 0.0
                self.annual_oil_production[y] = 0.0
        
        actual_gas_production = sum(self.annual_gas_production.values())
        actual_oil_production = sum(self.annual_oil_production.values())
        
        self.total_gas_production = actual_gas_production
        self.total_oil_production = actual_oil_production
        
        if output:
            print(f"\n[1. Production Adjustment]")
            print(f"  Original Gas: {original_gas_production:.2f} BCF")
            print(f"  Original Oil: {original_oil_production:.2f} MMbbls")
            print(f"  Actual Gas (to COP): {actual_gas_production:.2f} BCF")
            print(f"  Actual Oil (to COP): {actual_oil_production:.2f} MMbbls")
            print(f"  Reduction: {original_gas_production - actual_gas_production:.2f} BCF")
            print(f"  Reduction: {original_oil_production - actual_oil_production:.2f} MMbbls")
        
        # ===== 2. 수익 재계산 (생산량 조정 반영) =====
        for y in years:
            oil_vol = self.annual_oil_production.get(y, 0.0)
            gas_vol = self.annual_gas_production.get(y, 0.0)
            oil_price = self.oil_price_by_year.get(y, 0.0)
            gas_price = self.gas_price_by_year.get(y, 0.0)
            
            self.annual_revenue_oil[y] = oil_vol * oil_price
            self.annual_revenue_gas[y] = gas_vol * gas_price
            self.annual_revenue[y] = self.annual_revenue_oil[y] + self.annual_revenue_gas[y]
        
        # 누적 수익 재계산
        cumulative_revenue = 0.0
        for y in years:
            cumulative_revenue += self.annual_revenue.get(y, 0.0)
            self.annual_cum_revenue[y] = cumulative_revenue
        
        if output:
            print(f"\n[2. Revenue Recalculation]")
            print(f"  Total Revenue (adjusted): {sum(self.annual_revenue.values()):.2f} MM$")
        
        # ===== 3. OPEX 조정 (COP 이후 0) =====
        for y in years:
            if y > cop_year:
                self.annual_opex[y] = 0.0
                self.annual_opex_inflated[y] = 0.0
        
        # 누적 OPEX 재계산
        cumulative_opex = 0.0
        for y in years:
            cumulative_opex += self.annual_opex_inflated.get(y, 0.0)
            self.annual_cum_opex_inflated[y] = cumulative_opex
        
        if output:
            print(f"\n[3. OPEX Adjustment]")
            print(f"  Total OPEX (to COP): {sum(self.annual_opex_inflated.values()):.2f} MM$")
        
        # ===== 4. ABEX 조정 =====
        original_total_abex = sum(self.annual_abex_inflated.values())
        
        # 매장량의 절반 이상 생산된 시점 찾기
        cumulative_gas = 0.0
        abex_start_year = None
        half_reserve = original_gas_production / 2.0
        
        for y in sorted(years):
            # 원래 생산 계획 기준으로 절반 시점 계산
            if hasattr(self, '_original_gas_production'):
                cumulative_gas += self._original_gas_production.get(y, 0.0)
            else:
                cumulative_gas += self.annual_gas_production.get(y, 0.0)
            
            if cumulative_gas >= half_reserve:
                abex_start_year = y
                break
        
        # ABEX 기간의 실제 생산량 계산
        if abex_start_year and cop_year >= abex_start_year:
            abex_period_production = {}
            total_abex_period_production = 0.0
            
            for y in years:
                if abex_start_year <= y <= cop_year:
                    prod = self.annual_gas_production.get(y, 0.0)
                    abex_period_production[y] = prod
                    total_abex_period_production += prod
            
            # 실제 생산 비율에 따른 ABEX 조정
            production_ratio = actual_gas_production / original_gas_production if original_gas_production > 0 else 1.0
            adjusted_total_abex = original_total_abex * production_ratio
            
            # ABEX를 생산량 비율로 재분배
            adjusted_annual_abex_inflated = {}
            for y in years:
                if y < abex_start_year:
                    adjusted_annual_abex_inflated[y] = 0.0
                elif y <= cop_year and total_abex_period_production > 0:
                    prod_ratio = abex_period_production.get(y, 0.0) / total_abex_period_production
                    adjusted_annual_abex_inflated[y] = adjusted_total_abex * prod_ratio
                else:  # y > cop_year
                    adjusted_annual_abex_inflated[y] = 0.0
            
            self.annual_abex_inflated = adjusted_annual_abex_inflated
            
            if output:
                print(f"\n[4. ABEX Adjustment]")
                print(f"  ABEX Start Year: {abex_start_year}")
                print(f"  Original Total ABEX: {original_total_abex:.2f} MM$")
                print(f"  Adjusted Total ABEX: {sum(adjusted_annual_abex_inflated.values()):.2f} MM$")
                print(f"  Production Ratio: {production_ratio:.2%}")
        else:
            # ABEX가 시작되기 전에 COP 발생
            for y in years:
                self.annual_abex_inflated[y] = 0.0
            
            if output:
                print(f"\n[4. ABEX Adjustment]")
                print(f"  COP occurred before ABEX period - No ABEX")
        
        # 누적 ABEX 재계산
        cumulative_abex = 0.0
        for y in years:
            cumulative_abex += self.annual_abex_inflated.get(y, 0.0)
            self.annual_cum_abex_inflated[y] = cumulative_abex
        
        # ===== 5. Other fees 조정 (COP 이후 0) =====
        for y in years:
            if y > cop_year:
                self.other_fees[y] = 0.0
        
        # ===== 6. 감가상각 조정 (COP 이후 0) =====
        if self.annual_depreciation:
            for y in years:
                if y > cop_year:
                    self.annual_depreciation[y] = 0.0
        
        if output:
            print(f"\n[COP Adjustments Complete]")
            print(f"{'='*60}\n")
        
        return {
            'cop_year': cop_year,
            'original_gas_production': original_gas_production,
            'actual_gas_production': actual_gas_production,
            'original_abex': original_total_abex,
            'adjusted_abex': sum(self.annual_abex_inflated.values())
        }
    
    def calculate_royalty(self, output=False):
        """
        조광료 계산 - 반드시 apply_cop_adjustments() 이후에 호출
        """
        if not self.annual_revenue:
            raise ValueError("수익을 먼저 계산해야 합니다")
        
        self._build_full_timeline()
        
        cop_year = self.cop_year if self.cop_year else self.all_years[-1]
        years = self.all_years
        
        annual_r_factor = {}
        annual_royalty = {}
        cum_royalty = 0.0
        
        for y in years:
            # COP 이후에는 조광료 없음
            if y > cop_year:
                annual_r_factor[y] = 0.0
                annual_royalty[y] = 0.0
                continue
            
            # R-factor 계산 (조정된 누적값 사용)
            cum_rev_after = self.annual_cum_revenue.get(y, 0.0) - cum_royalty
            denom = (self.annual_cum_capex_inflated.get(y, 0.0) 
                    + self.annual_cum_opex_inflated.get(y, 0.0) 
                    + self.annual_cum_abex_inflated.get(y, 0.0))
            
            annual_r_factor[y] = cum_rev_after / denom if denom != 0 else 0.0
            rate = self._calculate_royalty_rates(annual_r_factor[y])
            annual_royalty[y] = self.annual_revenue.get(y, 0.0) * rate
            cum_royalty += annual_royalty[y]
        
        self.annual_r_factor = annual_r_factor
        self.annual_royalty = annual_royalty
        
        if output:
            print(f"[Royalty] Total Royalty: {cum_royalty:.2f} MM$")
        
        return self.annual_royalty
    
    def calculate_taxes(self, 
                        investment_tax_credit: bool = True, # 투자세액감면 대상 여부
                        local_tax: bool = True, # 지방세 포함 여부
                        output=False):
        """
        세금 계산 - 반드시 apply_cop_adjustments()와 calculate_royalty() 이후에 호출
        """
        self._build_full_timeline()
        
        if not self.annual_revenue:
            raise ValueError("수익을 먼저 계산해야 합니다")
        if not self.annual_royalty:
            raise ValueError("조광료를 먼저 계산해야 합니다")
        if not self.annual_depreciation:
            self.calculate_depreciation(output=False)
        if investment_tax_credit:
            self._calculate_investment_tax_credit()

        cop_year = self.cop_year if self.cop_year else self.all_years[-1]
        
        running_loss = 0.0                  # 이월결손금
        self.loss_carryforward = {}
        tax_credits_carried_forward = 0.0        # 세액공제
        self.tax_credits_carried_forward = {}

        for y in self.all_years:
            # COP 이후에는 세금 없음
            if y > cop_year:
                self.taxable_income[y] = 0.0
                self.loss_carryforward[y] = 0.0
                self.corporate_income_tax[y] = 0.0
                self.annual_total_tax[y] = 0.0
                continue
            
            self.loss_carryforward[y] = running_loss
            
            # 과세 표준 계산 (조정된 값들 사용)
            pre_tax_income = (self.annual_revenue.get(y, 0.0) 
                             - self.annual_royalty.get(y, 0.0)
                             - self.annual_opex_inflated.get(y, 0.0)
                             - self.annual_abex_inflated.get(y, 0.0)
                             - self.annual_depreciation.get(y, 0.0))
            
            # 이월결손금 처리
            taxable = 0.0
            if pre_tax_income < 0:
                running_loss += (-pre_tax_income)
                taxable = 0.0
            else:
                if running_loss > 0:
                    if pre_tax_income >= running_loss:
                        taxable = pre_tax_income - running_loss
                        running_loss = 0.0
                    else:
                        taxable = 0.0
                        running_loss -= pre_tax_income
                else:
                    taxable = pre_tax_income
            
            self.taxable_income[y] = taxable
            
            # 법인세 계산
            corp_tax = self._calculate_CIT(taxable)
            
            # 지방세 포함 여부
            if local_tax:
                corp_tax *= 1.1
            self.corporate_income_tax[y] = corp_tax

            # 세액 공제 적용 (통합투자세액공제)
            tax_credit = 0.0
            if investment_tax_credit:
                # 통합투자세액공제 계산
                tax_credit = self.annual_investment_tax_credit.get(y, 0.0)
                tax_credits_carried_forward = self.tax_credits_carried_forward.get(y, 0.0)

                # 투자세액공제 이월
                if corp_tax < tax_credit:
                    tax_credits_carried_forward += (tax_credit - corp_tax)
                    utilized_credit = corp_tax
                    tax_after_credit = 0
                elif corp_tax < (tax_credit+tax_credits_carried_forward):
                    tax_credits_carried_forward = (tax_credit + tax_credits_carried_forward) - corp_tax
                    utilized_credit = corp_tax
                    tax_after_credit = 0
                else:
                    # 사용 가능한 모든 공제액(당해연도 + 이월분)을 사용하고도 세금이 남는 경우
                    total_available = tax_credit + tax_credits_carried_forward
                    tax_after_credit = corp_tax - total_available
                    utilized_credit = total_available
                    tax_credits_carried_forward = 0

                # 농어촌특별세 (Rural Development Tax) 계산: 
                # 통합투자세액공제에 부과되며, 감면받은 세액의 20% 부과
                rd_tax = utilized_credit * 0.2
                
                self.annual_rural_development_tax[y] = rd_tax
                self.utilized_tax_credit[y] = utilized_credit
                self.annual_total_tax[y] = tax_after_credit + rd_tax
                self.tax_credits_carried_forward[y+1] = tax_credits_carried_forward
            else:
                self.annual_total_tax[y] = corp_tax
        
        self.total_tax = sum(self.annual_total_tax.values())
        
        if output:
            print(f"[Tax] Total Tax: {self.total_tax:.2f} MM$")
        
        return self.annual_total_tax
    
    def calculate_net_cash_flow(self, discovery_bonus: Optional[float] = None, output=False):
        """
            최종 현금흐름 계산 - 모든 조정이 완료된 후 단순 집계만 수행
            반드시 다음 순서로 호출되어야 함:
            1. calculate_annual_revenue()
            2. determine_cop_year() - cop 결정시 royalty 계산이 필요하나, 함수 내에 이를 포함함.
            3. apply_cop_adjustments()
            4. calculate_royalty()
            5. calculate_high_price_royalty()
            6. calculate_depreciation()
            7. calculate_taxes()
            8. calculate_net_cash_flow() ← 여기
        """
        self._build_full_timeline()
        
        # 필수 계산 확인
        if not self.annual_revenue:
            raise ValueError("수익 계산이 필요합니다")
        if not self.annual_royalty:
            raise ValueError("조광료 계산이 필요합니다")
        if not self.annual_total_tax:
            raise ValueError("세금 계산이 필요합니다")
        
        years = self.all_years
        
        # 현금흐름 계산 (단순 집계)
        self.annual_net_cash_flow = {}
        self.cumulative_cash_flow = {}
        cum_ncf = 0.0
        
        for y in years:
            rev = self.annual_revenue.get(y, 0.0)
            royalty = self.annual_royalty.get(y, 0.0)
            high_price_royalty = self.annual_high_price_royalty.get(y, 0.0)
            capex = self.annual_capex_inflated.get(y, 0.0)
            opex = self.annual_opex_inflated.get(y, 0.0)
            abex = self.annual_abex_inflated.get(y, 0.0)
            tax = self.annual_total_tax.get(y, 0.0)
            other = self.other_fees.get(y, 0.0)
            
            # Discovery bonus
            bonus = 0.0
            if discovery_bonus and y == self.production_start_year:
                bonus = discovery_bonus
            
            # NCF = Revenue - Royalty - High Price Royalty - (CAPEX + OPEX + ABEX + Other) - Tax
            ncf = rev - royalty - high_price_royalty - (capex + opex + abex + bonus + other) - tax
            
            self.annual_net_cash_flow[y] = ncf
            cum_ncf += ncf
            self.cumulative_cash_flow[y] = cum_ncf
        
        # 총합 계산
        self.total_revenue = sum(self.annual_revenue.values())
        self.total_royalty = sum(self.annual_royalty.values())
        self.total_capex = sum(self.annual_capex_inflated.values())
        self.total_opex = sum(self.annual_opex_inflated.values())
        self.total_abex = sum(self.annual_abex_inflated.values())
        self.total_tax = sum(self.annual_total_tax.values())
        
        if output:
            print(f"\n{'='*60}")
            print(f"[Net Cash Flow Summary]")
            print(f"{'='*60}")
            print(f"  Total Revenue:    {self.total_revenue:>12,.2f} MM$")
            print(f"  Total Royalty:    {self.total_royalty:>12,.2f} MM$")
            print(f"  Total CAPEX:      {self.total_capex:>12,.2f} MM$")
            print(f"  Total OPEX:       {self.total_opex:>12,.2f} MM$")
            print(f"  Total ABEX:       {self.total_abex:>12,.2f} MM$")
            print(f"  Total Tax:        {self.total_tax:>12,.2f} MM$")
            print(f"  {'-'*60}")
            print(f"  Net Cash Flow:    {cum_ncf:>12,.2f} MM$")
            print(f"{'='*60}\n")
        
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

    def get_annual_cash_flow_table(self, capex_detail=True, tax_detail=True):
        # 1. 컬럼 데이터 구성
        cols = [
            self.annual_oil_production, self.oil_price_by_year, 
            self.annual_gas_production, self.gas_price_by_year, 
            self.annual_revenue, self.annual_r_factor, self.annual_royalty, self.annual_high_price_royalty,
            self.annual_capex_inflated, 
        ]
        
        # 2. 인덱스 구성
        idx = [
            '석유 (MMbbl)', '유가 ($/bbl)', 
            '가스 (BCF)', '가스가 ($/mcf)', 
            '매출액 (MM$)', 'R-Factor', '조광료 (MM$)', '고유가 추가 조광료 (MM$)',
            'CAPEX (MM$)',
        ]

        if capex_detail:
            # Inflated breakdown을 사용
            breakdown_keys = [
                ('exploration', 'CAPEX - 탐사비 (MM$)'),
                ('drilling', 'CAPEX - 시추비 (MM$)'),
                ('subsea', 'CAPEX - 해저설비 (MM$)'),
                ('FPSO', 'CAPEX - FPSO (MM$)'),
                ('export_pipeline', 'CAPEX - Pipeline (MM$)'),
                ('terminal', 'CAPEX - Terminal (MM$)'),
                ('PM_others', 'CAPEX - PM/기타 (MM$)')
            ]
            for key, label in breakdown_keys:
                cols.append(self.capex_breakdown_inflated.get(key, {}))
                idx.append(label)

        cols.append(self.annual_opex_inflated)
        idx.append('OPEX (MM$)')
        cols.append(self.annual_abex_inflated)
        idx.append('ABEX (MM$)')
        cols.append(self.other_fees)
        idx.append('기타비용 (MM$)')

        if tax_detail:
            tax_cols = [
                (self.corporate_income_tax, '법인세 (MM$)'),
                (self.loss_carryforward, '이월결손금 공제 (MM$)'),
                (self.annual_investment_tax_credit, '투자세액감면 (MM$)'),
                (self.tax_credits_carried_forward, '투자세액감면 이월분 (MM$)'),
                (self.utilized_tax_credit, '투자세액감면 활용분 (MM$)'),
                (self.annual_rural_development_tax, '농어촌특별세 (MM$)')
            ]
            for data, label in tax_cols:
                cols.append(data)
                idx.append(label)

        cols.append(self.annual_total_tax)
        idx.append('세금 총액 (MM$)')
        cols.append(self.annual_net_cash_flow)
        idx.append('NCF (MM$)')

        df = pd.DataFrame(cols, index=idx)
        df = df.reindex(sorted(df.columns), axis=1)
        # NCF (MM$) 행의 값이 0이거나 NaN인 컬럼 제거
        df = df.loc[:, (df.loc['NCF (MM$)'] != 0) & (df.loc['NCF (MM$)'].notna())]
        df.insert(0, 'Total', df.sum(axis=1))
        return df

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