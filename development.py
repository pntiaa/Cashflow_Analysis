import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional, List # Ensure List is imported

# decorator for rounding
def rounding(func):
    def wrapper(**kwargs) -> Dict[int, float]:
        price_by_year = func(**kwargs)
        if not isinstance(price_by_year, dict):
            raise TypeError("The decorated function must return Dict[int, float].")
        return {year: round(price, 2) for year, price in price_by_year.items()}
    return wrapper

class DevelopmentCost:
    def __init__(self,
                 dev_start_year,
                 dev_param: Optional[Dict] = None, development_case: str = 'FPSO_case'):
        """
        dev_param: parameters dictionary, e.g.
          {
            'FPSO_case': {
               'drilling_cost': 10.0,         # MM$ per well
               'Subsea_cost': 2.0,            # MM$ per well
               'OPEX_per_well': 0.1,          # MM$ per well per year
               'ABEX_per_well': 0.05,         # MM$ per well (total)
               'ABEX_FPSO': 1.0,              # MM$
               'feasability_study': 0.2,      # MM$
               'concept_study_cost': 0.1,     # MM$
               'FEED_cost': 0.5,              # MM$
               'EIA_cost': 0.05,              # MM$
               'FPSO_cost': 50.0,             # MM$ (lump)
               'export_pipeline_cost': 10.0,  # MM$
               'terminal_cost': 2.0,          # MM$
               'PM_others_cost': 1.0,         # MM$
            }
          }
        """
        self.dev_param = dev_param or {}
        self.development_case = development_case
        self.case_param = self.dev_param.get(development_case, {})

        # schedule / time
        # self.annual_production: Dict[int, float] = {}
        self.annual_gas_production: Dict[int, float] = {}
        self.annual_oil_production: Dict[int, float] = {}
        self.total_gas_production: float = 0.0
        self.total_oil_production: float = 0.0

        self.yearly_drilling_schedule: Dict[int, int] = {}
        self.cost_years: List[int] = []          # sorted list of development years (may include year 0 if used)
        self.production_years: List[int] = [] # This attribute holds the list of actual production years

        self.drill_start_year: [int] = None     # 생산정 시추를 시작하는 년도
        self.dev_start_year: [int] = dev_start_year    # FPSO 등의 건조에 걸리는 시간을 감안하여 별도로 설정

        self.total_development_years: int = 0
        self.cumulative_well_count: Dict[int, int] = {}  # cumulative wells at each year (end of year)
        self._total_production_duration: Optional[int] = None # New attribute to store production duration

        # Annual dicts (all MM$ units)
        self.sunk_cost: float = 0.0
        self.exploration_costs: Dict[int, float] = {} #Added
        self.drilling_costs: Dict[int, float] = {}
        self.subsea_costs: Dict[int, float] = {}
        self.feasability_study_cost: Dict[int, float] = {}
        self.concept_study_cost: Dict[int, float] = {}
        self.FEED_cost: Dict[int, float] = {}
        self.EIA_cost: Dict[int, float] = {}

        self.FPSO_cost: Dict[int, float] = {}
        self.export_pipeline_cost: Dict[int, float] = {}
        self.terminal_cost: Dict[int, float] = {}
        self.PM_others_cost: Dict[int, float] = {}

        self.annual_capex: Dict[int, float] = {}
        self.annual_opex: Dict[int, float] = {}
        self.annual_abex: Dict[int, float] = {}
        self.total_annual_costs: Dict[int, float] = {}
        self.cumulative_costs: Dict[int, float] = {}

        # output scalars
        self.total_capex: float = 0.0
        self.total_opex: float = 0.0
        self.total_abex: float = 0.0

        # print(f"[init] DevelopmentCost initialized for: {development_case}")
        #if self.case_param:
        #    print(f"[init] Available cost parameters: {list(self.case_param.keys())}")

    # -----------------------
    # Utilities
    # -----------------------
    @staticmethod
    def _dict_zero_for_years(years: List[int]) -> Dict[int, float]:
        return {y: 0.0 for y in years}

    @staticmethod
    def _sum_dict_values(d: Dict[int, float]) -> float:
        return sum(d.values())

    @staticmethod
    def _rounding_dict_values(d: Dict[int, float]) -> Dict[int, float]:
        return {year: round(value, 2) for year, value in d.items()}
    # ---------------------------
    # 타임라인 구축 헬퍼 함수
    # ---------------------------
    def _build_full_timeline(self):
        """
        - 개발 및 생산 연도를 포함하는 전체 연도(`self.all_years`)를 구축.
        - 개발 딕셔너리를 정렬하고 누락된 부분을 0으로 채워 `self.annual_capex/opex/abex`를 채움.
        """
        years = set(self.cost_years)
        years |= set(self.oil_production_by_year.keys())
        years |= set(self.gas_production_by_year.keys())
        years |= set(self.annual_capex.keys())
        years |= set(self.annual_opex.keys())
        years |= set(self.annual_abex.keys())

        if len(years) == 0:
            raise ValueError("연도를 찾을 수 없습니다 (개발 또는 생산)")

    # -----------------------
    # Schedule setter
    # -----------------------
    def set_exploration_stage(self,
                              exploration_start_year: int = 2024,
                              exploration_costs: Dict[int, float] = None,
                              sunk_cost=None, output=True):
        self.exploration_costs = exploration_costs or {}
        years = set()
        if exploration_costs:
             max_year = max(exploration_costs.keys()) 
             years = set(range(int(exploration_start_year), int(max_year) + 1))
        
        years |= set(self.cost_years)
        years = sorted(list(years))

        for y in years:
            self.exploration_costs[y] = exploration_costs.get(y, 0.0)
        
        # add sunk cost on exploration_start_year
        self.exploration_costs[exploration_start_year] += sunk_cost
        if output:
            print(f"[exploration] exploration drilling ({self.exploration_costs.keys()})")
            print(f"[exploration] sunk cost ({self.sunk_cost} added on year {exploration_start_year})")

    def set_drilling_schedule(self, drill_start_year, yearly_drilling_schedule: Dict[int, int], already_shifted=False, output=True):
        """
        yearly_drilling_schedule: dict {year: wells}
        This function sorts years, computes cumulative wells and sets related attributes.
        """
        self.drill_start_year = drill_start_year
        if not isinstance(yearly_drilling_schedule, dict):
            raise ValueError("yearly_drilling_schedule must be a dict {year: wells}")
        if self.drill_start_year < self.dev_start_year:
            raise ValueError(f"dev_start_year({drill_start_year}) should be later than dev_start_year({self.dev_start_year})")

        # make a shallow copy and sort years
        shift = 0 if already_shifted else self.drill_start_year
        if not yearly_drilling_schedule:
             if output:
                 print("[schedule] No drilling schedule provided.")
             return

        for y_idx, num_wells in yearly_drilling_schedule.items():
            self.yearly_drilling_schedule[y_idx + shift] = num_wells
        
        schedule_keys = sorted(list(self.yearly_drilling_schedule.keys()))
        if not schedule_keys:
            self.cost_years = [self.dev_start_year]
        else:
            # dev_start_year부터 마지막 시추년도까지의 범위를 생성
            self.cost_years = sorted(list(range(self.dev_start_year, schedule_keys[-1] + 1, 1)))
        
        if len(self.cost_years) == 0:
            self.cost_years = [self.dev_start_year]

        self.total_development_years = len(self.cost_years)

        # cumulative well count at end of each development year
        cum = 0
        self.cumulative_well_count = {}
        for y in self.cost_years:
            cum += int(self.yearly_drilling_schedule.get(y, 0))
            self.cumulative_well_count[y] = cum
        if output:
            # print(f"[schedule] Drilling schedule set ({self.total_development_years} years): {self.yearly_drilling_schedule}")
            print(f"[schedule] Drilling schedule set ({len([k for k, v in self.yearly_drilling_schedule.items() if v >0])} years): {self.yearly_drilling_schedule}")
            print(f"[schedule] Cumulative wells by year: {self.cumulative_well_count}")
            print(f"[schedule] Drill period: {self.drill_start_year} - {self.cost_years[-1]}")
            print(f"[schedule] Total wells: {self.cumulative_well_count[self.cost_years[-1]]}")

    def set_annual_production(self, annual_gas_production: Dict[int, float], annual_oil_production: Dict[int, float], already_shifted=False, output=True):
        shift = 0 if already_shifted else self.drill_start_year
        for y_idx, value in annual_gas_production.items():
            self.annual_gas_production[y_idx + shift] = value

        for y_idx, value in annual_oil_production.items():
            self.annual_oil_production[y_idx + shift] = value

        self.production_years = list(self.annual_gas_production.keys())
        # Calculate _total_production_duration based on years with production > 0
        self._total_production_duration = sum(1 for year, prod in annual_gas_production.items() if prod > 0)
        self.total_gas_production = self._sum_dict_values(self.annual_gas_production)
        self.total_oil_production = self._sum_dict_values(self.annual_oil_production)

        # cost_year 업데이트
        years = set(self.cost_years)
        years |= set(self.annual_gas_production.keys())
        years |= set(self.annual_oil_production.keys())
        self.cost_years = list(years)

        if output:
            print(f"[set_annual_production] Active production duration: {self._total_production_duration} years")
        # return self.annual_production
    # -----------------------
    # CAPEX components
    # -----------------------
    def calculate_drilling_costs(self, output=True) -> Dict[int, float]:
        if not self.yearly_drilling_schedule:
            raise ValueError("Drilling schedule not set. Call set_drilling_schedule() first.")
        if 'drilling_cost' not in self.case_param:
            raise ValueError("'drilling_cost' not found in case_param")

        cost_per_well = float(self.case_param['drilling_cost'])
        self.drilling_costs = {y: int(self.yearly_drilling_schedule.get(y, 0)) * cost_per_well for y in self.cost_years}
        if output:
            # print(f"[drilling] cost_per_well={cost_per_well} -> drilling_costs: {self.drilling_costs}")
            return self.drilling_costs

    def calculate_subsea_costs(self, output=True) -> Dict[int, float]:
        if not self.yearly_drilling_schedule:
            raise ValueError("Drilling schedule not set.")
        if 'Subsea_cost' not in self.case_param:
            raise ValueError("'Subsea_cost' not found in case_param")

        subsea_per_well = float(self.case_param['Subsea_cost'])
        self.subsea_costs = {y: int(self.yearly_drilling_schedule.get(y, 0)) * subsea_per_well for y in self.cost_years}
        if output:
             print(f"[subsea] subsea_per_well={subsea_per_well} -> subsea_costs: {self.subsea_costs}")
             return self.subsea_costs

    def calculate_study_costs(self, timing: str = 'year_0', output=True) -> Dict[int, float]:
        """
        timing:
          - 'year_0': all study costs in year_before_start (dev_start_year - 1)
          - 'year_1': all in first development year (dev_start_year)
          - 'spread': evenly across development years
        """
        if not self.yearly_drilling_schedule:
            raise ValueError("Drilling schedule not set.")

        feas = float(self.case_param.get('feasability_study', 0.0))
        concept = float(self.case_param.get('concept_study_cost', 0.0))
        feed = float(self.case_param.get('FEED_cost', 0.0))
        eia = float(self.case_param.get('EIA_cost', 0.0))

        # prepare years for study cost dict depending on timing
        if timing == 'year_0':
            study_years = [self.dev_start_year - 1] + self.cost_years
        else:
            study_years = self.cost_years.copy()

        # initialize zero dicts for study components
        self.feasability_study_cost = self._dict_zero_for_years(study_years)
        self.concept_study_cost = self._dict_zero_for_years(study_years)
        self.FEED_cost = self._dict_zero_for_years(study_years)
        self.EIA_cost = self._dict_zero_for_years(study_years)

        if timing == 'year_0':
            self.feasability_study_cost[self.dev_start_year - 1] = feas
            self.concept_study_cost[self.dev_start_year - 1] = concept
            self.FEED_cost[self.dev_start_year - 1] = feed
            self.EIA_cost[self.dev_start_year - 1] = eia
        elif timing == 'year_1':
            self.feasability_study_cost[self.dev_start_year] = feas
            self.concept_study_cost[self.dev_start_year] = concept
            self.FEED_cost[self.dev_start_year] = feed
            self.EIA_cost[self.dev_start_year] = eia
        elif timing == 'spread':
            n = max(1, self.total_development_years)
            per_feas = feas / n
            per_concept = concept / n
            per_feed = feed / n
            per_eia = eia / n
            for y in self.cost_years:
                self.feasability_study_cost[y] = per_feas
                self.concept_study_cost[y] = per_concept
                self.FEED_cost[y] = per_feed
                self.EIA_cost[y] = per_eia
        else:
            raise ValueError("Unknown timing for study_costs. Use 'year_0', 'year_1' or 'spread'.")

        total = feas + concept + feed + eia
        if output:
            print(f"[study] timing={timing}, total_study_cost={total} -> study : {self.feasability_study_cost}")
            # print(f"[study] timing={timing}, total_study_cost={total} -> study dict keys: {list(self.feasability_study_cost.keys())}")
            return {
                'feasability': self.feasability_study_cost,
                'concept': self.concept_study_cost,
                'FEED': self.FEED_cost,
                'EIA': self.EIA_cost
            }

    def calculate_facility_costs(self, timing: str = 'year_1', output=True) -> Dict[str, Dict[int, float]]:
        """
        timing: 'year_1' (all in first development year) or 'spread' (spread evenly across development years)
        """
        if not self.yearly_drilling_schedule:
            raise ValueError("Drilling schedule not set.")

        FPSO = float(self.case_param.get('FPSO_cost', 0.0))
        pipeline = float(self.case_param.get('export_pipeline_cost', 0.0))
        terminal = float(self.case_param.get('terminal_cost', 0.0))
        pm = float(self.case_param.get('PM_others_cost', 0.0))

        if timing == 'year_1':
            self.FPSO_cost = self._dict_zero_for_years(self.cost_years)
            self.export_pipeline_cost = self._dict_zero_for_years(self.cost_years)
            self.terminal_cost = self._dict_zero_for_years(self.cost_years)
            self.PM_others_cost = self._dict_zero_for_years(self.cost_years)

            first = self.dev_start_year
            self.FPSO_cost[first] = FPSO
            self.export_pipeline_cost[first] = pipeline
            self.terminal_cost[first] = terminal
            self.PM_others_cost[first] = pm
        elif timing == 'spread':
            n = max(1, self.total_development_years)
            per_f = FPSO / n
            per_p = pipeline / n
            per_t = terminal / n
            per_m = pm / n
            self.FPSO_cost = {y: per_f for y in self.cost_years}
            self.export_pipeline_cost = {y: per_p for y in self.cost_years}
            self.terminal_cost = {y: per_t for y in self.cost_years}
            self.PM_others_cost = {y: per_m for y in self.cost_years}
        else:
            raise ValueError("Unknown timing for facility_costs. Use 'year_1' or 'spread'.")

        if output:
            print(f"[facility] timing={timing}. FPSO:{FPSO}, pipeline:{pipeline}, terminal:{terminal}, PM:{pm}")
            return {
                'FPSO': self.FPSO_cost,
                'export_pipeline': self.export_pipeline_cost,
                'terminal': self.terminal_cost,
                'PM_others': self.PM_others_cost
            }

    # -----------------------
    # OPEX / ABEX
    # -----------------------
    def calculate_annual_opex(self, study_timing: str = 'year_0', output=True) -> Dict[int, float]:
        """
        Returns annual_opex as {year: opex}
        OPEX logic: Calculated based on annual production and fixed OPEX.
        """
        if not self.yearly_drilling_schedule:
            raise ValueError("Drilling schedule not set.")
        if not self.annual_gas_production:
            raise ValueError("Annual Production is not set.")
        if 'OPEX_per_bcf' not in self.case_param:
            raise ValueError("'OPEX_per_bcf' not in case_param")
        if 'OPEX_fixed' not in self.case_param:
            raise ValueError("'OPEX_fixed' not in case_param")

        # opex_per_well = float(self.case_param['OPEX_per_well'])
        opex_fixed = float(self.case_param['OPEX_fixed'])
        opex_per_bcf = float(self.case_param['OPEX_per_bcf'])
        annual_opex: Dict[int, float] = {}

        # Iterate directly over the items of self.annual_production to get correct years and gas volumes
        # Add zero OPEX for development years if they are not in annual_production
        for year, gas_prod_bcf in self.annual_gas_production.items():
            if gas_prod_bcf == 0:
                annual_opex[year] = 0
            else:
                annual_opex[year] = (gas_prod_bcf * opex_per_bcf) + opex_fixed

        # The study_timing logic is to ensure the self.cost_years list also covers study_timing == 'year_0'
        # For OPEX calculation during development, we should consider all relevant years.
        all_relevant_years = sorted(list(set(self.cost_years) | set(self.annual_gas_production.keys())))
        # apply rounding
        annual_opex = self._rounding_dict_values(annual_opex)
        self.annual_opex = dict(sorted(annual_opex.items()))
        if output:
            print(f"[opex] OPEX_per_bcf={opex_per_bcf:,.2f}, [opex] OPEX_fixed ={opex_fixed:,.2f}. annual_opex keys: {list(self.annual_opex.keys())}")
            return self.annual_opex

    def calculate_annual_abex(self, study_timing: str = 'year_0', output=True) -> Dict[int, float]:
        """
        Simple ABEX handling: total ABEX (per well + FPSO/subsea/pipeline) is booked in the last year
        of the whole timeline (development + production).
        """
        if not self.yearly_drilling_schedule:
            raise ValueError("Drilling schedule not set.")
        # Changed: _total_production_duration is now calculated in set_annual_production.
        # If annual_production was not set, this would be None, but it should be set by set_annual_production.
        if self._total_production_duration is None:
             # Fallback if somehow annual_production was empty or all values were zero
            print("[abex] Warning: _total_production_duration is None. Using default for ABEX fallback.")
            total_project_duration_for_abex_calc = len(self.production_years) if self.production_years else 0
        else:
            total_project_duration_for_abex_calc = self._total_production_duration

        abex_per_well = float(self.case_param.get('ABEX_per_well', 0.0))
        abex_FPSO = float(self.case_param.get('ABEX_FPSO', 0.0))
        abex_subsea = float(self.case_param.get('ABEX_subsea', 0.0))
        abex_onshore = float(self.case_param.get('ABEX_onshore_pipeline', 0.0))
        abex_offshore = float(self.case_param.get('ABEX_offshore_pipeline', 0.0))

        total_wells = sum(self.yearly_drilling_schedule.values())
        total_abex = abex_per_well * total_wells + abex_FPSO + abex_subsea + abex_onshore + abex_offshore

        # construct year keys (include possible year_0)
        dev_years = self.cost_years.copy()
        if study_timing == 'year_0':
            years = [self.dev_start_year - 1] + dev_years
        else:
            years = dev_years.copy()

        # The ABEX should be booked in the last year of the entire project, which is derived from the maximum of
        # all development years and all production years.
        all_project_years = sorted(list(set(years) | set(self.annual_gas_production.keys())))
        if all_project_years:
            actual_last_project_year = all_project_years[-1]
        else:
            # Fallback if no development or production years for some reason
            actual_last_project_year = self.drill_start_year + total_project_duration_for_abex_calc - 1

        # Initialize annual_abex for all years up to actual_last_project_year
        all_abex_years = sorted(list(set(all_project_years) | set(years)))
        annual_abex = {y: 0.0 for y in all_abex_years}

        annual_abex[actual_last_project_year] = total_abex

        # cost years update
        years = set(self.cost_years)
        years |= set(self.exploration_costs.keys())
        years |= set(self.annual_gas_production.keys())
        years |= set(self.annual_oil_production.keys())
        years |= set(self.annual_capex.keys())
        years |= set(self.annual_abex.keys())
        self.cost_years = list(years)

        self.annual_abex = dict(sorted(annual_abex.items()))
        if output:
            print(f"[abex] total_abex={total_abex:,.2f} booked in year {actual_last_project_year}")
            return self.annual_abex

    # -----------------------
    # Total costs
    # -----------------------
    def calculate_total_costs(self, production_years: int = 30, study_timing: str = 'year_0', facility_timing: str = 'year_1', output=True) -> Dict[str, object]:
        """
        Calculate everything and populate annual_capex, annual_opex, annual_abex, total_annual_costs, cumulative_costs.
        Returns a output dict with totals and the annual dicts.
        """
        if not self.yearly_drilling_schedule:
            raise ValueError("Drilling schedule not set.")

        # This 'production_years' argument still useful as overall project length if annual_production not set or all zero
        # But _total_production_duration will reflect active production duration if set.
        # If self._total_production_duration is still None here, it means annual_production was not set or was all zeros.
        if self._total_production_duration is None: # This check is to handle cases where annual_production is empty or all zeros.
            self._total_production_duration = production_years # Fallback to the argument for overall project duration

        # CAPEX components
        self.calculate_drilling_costs(output=output)
        self.calculate_subsea_costs(output=output)
        self.calculate_study_costs(timing=study_timing, output=output)
        self.calculate_facility_costs(timing=facility_timing, output=output)

        # Build annual CAPEX dict (years depend on study_timing)
        dev_years = self.cost_years.copy()
        if study_timing == 'year_0':
            years = [self.dev_start_year - 1] + dev_years
        else:
            years = dev_years.copy()

        # Ensure all component dicts have the same keys (fill zeros where missing)
        def ensure_keys(d: Dict[int, float], keys: List[int]) -> Dict[int, float]:
            return {k: float(d.get(k, 0.0)) for k in keys}

        # Components for CAPEX
        drilling = ensure_keys(self.drilling_costs, years)
        subsea = ensure_keys(self.subsea_costs, years)
        fps = ensure_keys(self.FPSO_cost, years)
        export = ensure_keys(self.export_pipeline_cost, years)
        term = ensure_keys(self.terminal_cost, years)
        pm = ensure_keys(self.PM_others_cost, years)
        feas = ensure_keys(self.feasability_study_cost, years)
        conc = ensure_keys(self.concept_study_cost, years)
        feed = ensure_keys(self.FEED_cost, years)
        eia = ensure_keys(self.EIA_cost, years)
        explo = ensure_keys(self.exploration_costs if self.exploration_costs else {}, years)

        # sum CAPEX by year
        self.annual_capex = {y: drilling[y] + subsea[y] + fps[y] + export[y] + term[y] + pm[y] + feas[y] + conc[y] + feed[y] + eia[y] + explo[y] for y in years}
        self.total_capex = self._sum_dict_values(self.annual_capex)

        # OPEX and ABEX
        self.calculate_annual_opex(study_timing=study_timing, output=output)
        self.calculate_annual_abex(study_timing=study_timing, output=output)

        # Build total_annual_costs for full timeline (development + production)
        # This should now include all years from self.annual_opex and self.annual_abex
        all_cost_years = sorted(list(set(self.annual_capex.keys()) | set(self.annual_opex.keys()) | set(self.annual_abex.keys())))

        def get_val_safe(d: Dict[int, float], y: int) -> float:
            return float(d.get(y, 0.0))

        self.total_annual_costs = {y: get_val_safe(self.annual_capex, y) + get_val_safe(self.annual_opex, y) + get_val_safe(self.annual_abex, y) for y in all_cost_years}

        # cumulative
        cum = 0.0
        self.cumulative_costs = {}
        for y in sorted(self.total_annual_costs.keys()):
            cum += self.total_annual_costs[y]
            self.cumulative_costs[y] = cum

        # totals
        self.total_opex = self._sum_dict_values(self.annual_opex)
        self.total_abex = self._sum_dict_values(self.annual_abex)
        total_project_cost = self.total_capex + self.total_opex + self.total_abex

        # print output
        if output:
            print("="*50)
            print("[output]")
            print(f"Total CAPEX: {self.total_capex:10,.2f} MM$")
            print(f"Total OPEX:  {self.total_opex:10,.2f} MM$")
            print(f"Total ABEX:  {self.total_abex:10,.2f} MM$")
            print(f"TOTAL PROJECT COST: {total_project_cost:10,.2f} MM$")
            print("="*50)

            return {
                'annual_capex': dict(sorted(self.annual_capex.items())),
                'annual_opex': dict(sorted(self.annual_opex.items())),
                'annual_abex': dict(sorted(self.annual_abex.items())),
                'total_annual_costs': dict(sorted(self.total_annual_costs.items())),
                'cumulative_costs': dict(sorted(self.cumulative_costs.items())),
                'total_capex': self.total_capex,
                'total_opex': self.total_opex,
                'total_abex': self.total_abex,
                'total_project_cost': total_project_cost
            }

    def get_cost_breakdown(self) -> Dict[str, object]:
        """
        Returns a breakdown of CAPEX, OPEX, and ABEX components.
        """
        all_years = sorted(list(set(self.annual_capex.keys()) | set(self.annual_opex.keys()) | set(self.annual_abex.keys())))
        
        # Ensure all components cover the full timeline
        def ensure_keys(d: Dict[int, float], keys: List[int]) -> Dict[int, float]:
            return {k: float(d.get(k, 0.0)) for k in keys}

        capex_breakdown = {
            'exploration': ensure_keys(self.exploration_costs if self.exploration_costs else {}, all_years),
            'drilling': ensure_keys(self.drilling_costs, all_years),
            'subsea': ensure_keys(self.subsea_costs, all_years),
            'FPSO': ensure_keys(self.FPSO_cost, all_years),
            'export_pipeline': ensure_keys(self.export_pipeline_cost, all_years),
            'terminal': ensure_keys(self.terminal_cost, all_years),
            'PM_others': ensure_keys(self.PM_others_cost, all_years),
            'feasibility_study': ensure_keys(self.feasability_study_cost, all_years),
            'concept_study': ensure_keys(self.concept_study_cost, all_years),
            'FEED': ensure_keys(self.FEED_cost, all_years),
            'EIA': ensure_keys(self.EIA_cost, all_years),
        }
        
        return {
            'capex_breakdown': capex_breakdown,
            'annual_opex': self.annual_opex,
            'annual_abex': self.annual_abex
        }


class QuestorDevelopmentCost(DevelopmentCost):
    def __init__(self, dev_start_year: int, excel_file_path: str, sheet_name: str):
        # 부모 클래스 초기화
        super().__init__(dev_start_year=dev_start_year, dev_param={}, development_case='QUE$TOR_Final')
        self.questor_raw: Dict = {}  # 로딩된 원본 데이터를 확인할 수 있도록 속성 추가
        self.drill_start_year = dev_start_year # Initialize for serialization compatibility
        self.load_questor_file(excel_file_path, sheet_name)

    def _find_keyword(self, df: pd.DataFrame, keyword: str, row_idx: Optional[int] = None, col_idx: Optional[int] = None):
        """특정 키워드가 있는 (행, 열) 인덱스를 반환 (특정 행 또는 열 지정 가능)"""
        target = keyword.strip().upper()
        
        if row_idx is not None:
            mask = df.iloc[row_idx, :].apply(lambda x: str(x).strip().upper() == target)
            return (row_idx, np.where(mask)[0][0]) if mask.any() else (None, None)
        
        if col_idx is not None:
            mask = df.iloc[:, col_idx].apply(lambda x: str(x).strip().upper() == target)
            return (np.where(mask)[0][0], col_idx) if mask.any() else (None, None)

        mask = df.map(lambda x: str(x).strip().upper() == target)
        if mask.any().any():
            r, c = np.where(mask)
            return r[0], c[0]
        return None, None

    def _generate_headers(self, header_df: pd.DataFrame) -> List[str]:
        """다중 행 헤더에서 숫자를 제외하고 가장 아래쪽 텍스트를 선택하여 헤더 생성"""
        new_headers = []
        for c in range(header_df.shape[1]):
            col_data = header_df.iloc[:, c].dropna().astype(str).tolist()
            # 숫자가 포함된 데이터 제외 및 공백 제거
            filtered = [t.strip() for t in col_data if not any(char.isdigit() for char in t) and t.strip()]
            
            # 상하 텍스트 중 하단 텍스트만 선택 (결합 로직 대체)
            if filtered:
                title = filtered[-1]
            else:
                title = f"Unknown_{c}"
            new_headers.append(title)
            
        if new_headers:
            new_headers[0] = "Year" # 첫 열 강제 지정
        return new_headers

    def _get_sum_of_columns(self, q_dict: Dict, keywords: List[str], category: str = "") -> Dict[int, float]:
            """합산 로직 디버깅 메시지 추가"""
            if not q_dict: return {}

            any_key = list(q_dict.keys())[0]
            combined_values = {y: 0.0 for y in q_dict[any_key].keys()}
            found_cols = []

            for col_name, yearly_data in q_dict.items():
                if any(kw.upper() in col_name.upper() for kw in keywords):
                    # ❗주의: 이미지상 'Total' 열이 Opex 합계를 이미 가지고 있다면 상세 항목 합산과 충돌할 수 있음
                    # 만약 상세 항목만 더하고 싶다면 "TOTAL" 제외 로직 유지
                    if "TOTAL" not in col_name.upper():
                        for y, val in yearly_data.items():
                            combined_values[y] += val
                        found_cols.append(col_name)

            if category:
                print(f"  [{category}] 매핑된 컬럼: {found_cols}")
            return combined_values

    def load_questor_file(self, excel_file_path: str, sheet_name: str):
        """전체 로드 프로세스"""
        try:
            # 1. 시트 전체 로드 (메모 포함)
            full_df = pd.read_excel(excel_file_path, sheet_name=sheet_name, header=None)

            # 2. 주요 앵커 위치 탐색
            anchor_row, anchor_col = self._find_keyword(full_df, "Year")
            # Year와 동일한 열(Column)에서 Total을 찾음
            total_row, _ = self._find_keyword(full_df, "TOTAL", col_idx=anchor_col)

            if anchor_row is None or total_row is None:
                raise ValueError("필수 키워드(Year 또는 TOTAL)를 시트에서 찾을 수 없습니다.")

            # 3. 유효 열 범위 확정 (A열부터 데이터가 끝나는 열까지)
            valid_cols_mask = full_df.iloc[total_row].notna() # anchor row에서 찾으면 year에서 찾으니깐, total row에서 찾기
            last_col_idx = valid_cols_mask[valid_cols_mask].index[-1]

            # 4. 헤더 생성 및 데이터 추출
            ## iloc으로 추출하면 마지막행은 추출되지 않으므로 last_col_idx+1까지 지정해야 전체 지정됨
            header_range = full_df.iloc[anchor_row : total_row, anchor_col : last_col_idx+1]
            cleaned_titles = self._generate_headers(header_range)
            data_part = full_df.iloc[total_row + 1 :, anchor_col : last_col_idx + 1].copy()
            data_part.columns = cleaned_titles

            # 5. 데이터 정제 (숫자 행만 남기기)
            data_part = (
                data_part[pd.to_numeric(data_part['Year'], errors='coerce').notna()]
                .set_index('Year')
                .apply(pd.to_numeric, errors='coerce')
                .fillna(0)
            )
            # 6. 'TOTAL' 열 제거
            data_part = data_part.drop(columns=[c for c in data_part.columns if c.upper() == 'TOTAL'], errors='ignore')

            # 로딩된 전체 데이터를 인스턴스 변수에 저장
            self.questor_raw = data_part.to_dict(orient='dict')

            # 6. 내부 속성 매핑
            self._set_annual_production(self.questor_raw)
            self._set_annual_costs(self.questor_raw)
            self.update_summary_metrics()

            print(f"✅ QUE$TOR 데이터 로드 성공: {len(data_part)}개 연도 감지")
            print(f"📌 확인 가능한 전체 컬럼 리스트: {list(self.questor_raw.keys())}")

        except Exception as e:
            print(f"❌ 로드 중 오류 발생: {e}")

    def set_exploration_stage(self, exploration_start_year: int = 2024, exploration_costs: Dict[int, float] = None, sunk_cost=None, output=True):
        super().set_exploration_stage(exploration_start_year, exploration_costs, sunk_cost, output)
        # Re-run cost aggregation to include exploration costs
        if self.questor_raw:
            self._set_annual_costs(self.questor_raw, output=False)
            self.update_summary_metrics()

    def _set_annual_costs(self, questor_dict: Dict, output=True):
        dev_start = self.dev_start_year
        # 키워드 기반 유연한 합산 매핑
        capex_raw = self._get_sum_of_columns(questor_dict, ['PROJECT', 'Facilities', 'Pipelines'], "CAPEX")
        opex_raw = self._get_sum_of_columns(questor_dict, ['OPEX','Fixed OPEX', 'Variable OPEX', 'Tariffs', 'Leases'], "OPEX")
        abex_raw = self._get_sum_of_columns(questor_dict, ['DECOMM','DECOMM.'], "ABEX")

        self.annual_capex = {} # Initialize clean
        for k, v in capex_raw.items(): self.annual_capex[k + dev_start] = v
        
        # Add exploration costs if they exist
        if self.exploration_costs:
            for k, v in self.exploration_costs.items():
                self.annual_capex[k] = self.annual_capex.get(k, 0.0) + v

        self.annual_opex = {}
        for k, v in opex_raw.items(): self.annual_opex[k + dev_start] = v
        
        self.annual_abex = {}
        for k, v in abex_raw.items(): self.annual_abex[k + dev_start] = v

        self.cost_years = sorted(set(self.annual_capex.keys()) | set(self.annual_opex.keys()) | set(self.annual_abex.keys()))

    def _set_annual_production(self, questor_dict: Dict, output=True):
            dev_start = self.dev_start_year
            gas_raw = self._get_sum_of_columns(questor_dict, ['Gas Bscf'], "GAS")
            oil_raw = self._get_sum_of_columns(questor_dict, ['Oil MMbbl', 'Cond. MMbbl'], "OIL")

            for k, v in gas_raw.items(): self.annual_gas_production[k + dev_start] = v
            for k, v in oil_raw.items(): self.annual_oil_production[k + dev_start] = v
            self.production_years = sorted(list(self.annual_gas_production.keys()))

    def update_summary_metrics(self):
        """매핑된 데이터를 기반으로 합계 지표 업데이트"""
        self.total_capex = sum(self.annual_capex.values())
        self.total_opex = sum(self.annual_opex.values())
        self.total_abex = sum(self.annual_abex.values())

        all_years = sorted(self.cost_years)
        self.total_annual_costs = {y: self.annual_capex.get(y, 0) + self.annual_opex.get(y, 0) + self.annual_abex.get(y, 0)
                                   for y in all_years}

        cum = 0.0
        self.cumulative_costs = {}
        for y in all_years:
            cum += self.total_annual_costs[y]
            self.cumulative_costs[y] = cum

if __name__ == "__main__":
    pass    