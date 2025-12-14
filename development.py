import numpy as np
import matplotlib.pyplot as plt
import copy
from typing import Dict, Optional, List # Ensure List is imported

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

    # -----------------------
    # Schedule setter
    # -----------------------
    def set_exploration_stage(self,
                              exploration_start_year: int = 2024,
                              exploration_costs: Dict[int, float] = None,
                              sunk_cost=None, output=True):
        self.exploration_costs = exploration_costs

        if output:
            print(f"[exploration] exploration drilling ({len(self.exploration_costs.keys())} years): {self.yearly_drilling_schedule}")
            print(f"[exploration] Cumulative wells by year: {self.cumulative_well_count}")


    def set_drilling_schedule(self, drill_start_year, yearly_drilling_schedule: Dict[int, int], output=True):
        """
        yearly_drilling_schedule: dict {year: wells}
        This function sorts years, computes cumulative wells and sets related attributes.
        """
        self.drill_start_year = drill_start_year
        if not isinstance(yearly_drilling_schedule, dict):
            raise ValueError("yearly_drilling_schedule must be a dict {year: wells}")
        if self.drill_start_year < self.dev_start_year:
            raise ValueError(f"dev_start_year({drill_start_year}) should be later than dev_start_year({dev_start_year})")

        # make a shallow copy and sort years
        for y_idx, num_wells in yearly_drilling_schedule.items():
            self.yearly_drilling_schedule[y_idx+self.drill_start_year] = num_wells
        # self.yearly_drilling_schedule = copy.deepcopy(yearly_drilling_schedule)

        # self.cost_years = sorted(list(self.yearly_drilling_schedule.keys())) dev_start_year가 더 빠를경우 오류생길 수 있음
        self.cost_years = sorted(list(range(self.dev_start_year, list(self.yearly_drilling_schedule.keys())[-1],1)))

        if len(self.cost_years) == 0:
            raise ValueError("yearly_drilling_schedule must contain at least one year")

        # self.drill_start_year = self.cost_years[0]
        self.total_development_years = len(self.cost_years)

        # cumulative well count at end of each development year
        cum = 0
        self.cumulative_well_count = {}
        for y in self.cost_years:
            cum += int(self.yearly_drilling_schedule.get(y, 0))
            self.cumulative_well_count[y] = cum
        if output:
            print(f"[schedule] Drilling schedule set ({self.total_development_years} years): {self.yearly_drilling_schedule}")
            print(f"[schedule] Cumulative wells by year: {self.cumulative_well_count}")
            print(f"[schedule] Drill period: {self.drill_start_year} - {self.cost_years[-1]}")
            print(f"[schedule] Total wells: {self.cumulative_well_count[self.cost_years[-1]]}")

    def set_annual_production(self, annual_gas_production: Dict[int, float], annual_oil_production: Dict[int, float], output=True):
        for y_idx, value in annual_gas_production.items():
            self.annual_gas_production[y_idx+self.drill_start_year] = value

        for y_idx, value in annual_oil_production.items():
            self.annual_oil_production[y_idx+self.drill_start_year] = value

        self.production_years = list(self.annual_gas_production.keys())
        # Calculate _total_production_duration based on years with production > 0
        self._total_production_duration = sum(1 for year, prod in annual_gas_production.items() if prod > 0)
        self.total_gas_production = self._sum_dict_values(self.annual_gas_production)
        self.total_oil_production = self._sum_dict_values(self.annual_oil_production)

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
            print(f"[study] timing={timing}, total_study_cost={total} -> study dict keys: {list(self.feasability_study_cost.keys())}")
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

        self.annual_abex = dict(sorted(annual_abex.items()))
        if output:
            print(f"[abex] total_abex={total_abex:,.2f} booked in year {actual_last_project_year}")
            return self.annual_abex

    # -----------------------
    # Total costs and plotting
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

        # sum CAPEX by year
        self.annual_capex = {y: drilling[y] + subsea[y] + fps[y] + export[y] + term[y] + pm[y] + feas[y] + conc[y] + feed[y] + eia[y] for y in years}
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

    def plot_cost_profile(self, show: bool = True):
        """
        Plot annual capex/opex/abex stacked bar and cumulative curve.
        Relies on self.total_annual_costs, self.annual_capex, self.annual_opex, self.annual_abex being present.
        """
        if not self.total_annual_costs:
            raise ValueError("Costs not calculated. Run calculate_total_costs() first.")

        years = sorted(self.total_annual_costs.keys())
        capex_vals = [self.annual_capex.get(y, 0.0) for y in years]
        opex_vals = [self.annual_opex.get(y, 0.0) for y in years]
        abex_vals = [self.annual_abex.get(y, 0.0) for y in years]
        total_vals = [self.total_annual_costs.get(y, 0.0) for y in years]
        cum_vals = [self.cumulative_costs.get(y, 0.0) for y in years]

        # Plot 1: stacked bars
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

        ax1.bar(years, capex_vals, label='CAPEX')
        ax1.bar(years, opex_vals, bottom=capex_vals, label='OPEX')
        bottom_for_abex = [c + o for c, o in zip(capex_vals, opex_vals)]
        ax1.bar(years, abex_vals, bottom=bottom_for_abex, label='ABEX')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Cost (MM$)')
        ax1.set_title(f'Annual Cost Profile - {self.development_case}')
        y_max = max(bottom_for_abex)*1.2
        y_max_digit = int(np.log10(y_max))
        y_max_grid = np.round(y_max, y_max_digit*-1)
        ax1.set_yticks(np.linspace(0,y_max_grid, 6))
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 3: drilled wells
        ax3 = ax1.twinx()
        # Align drilled_wells with the full timeline (years)
        drilled_wells_aligned = [self.yearly_drilling_schedule.get(y, 0) for y in years]
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
        if show:
            plt.show()

    def plot_total_annual_costs(self, show: bool = True):
        """
        Plots the total annual costs as a bar chart.
        """
        if not self.total_annual_costs:
            raise ValueError("Total annual costs have not been calculated. Call calculate_total_costs() first.")

        years = list(self.total_annual_costs.keys())
        costs = list(self.total_annual_costs.values())

        plt.figure(figsize=(12, 6))
        plt.bar(years, costs, color='skyblue')
        plt.xlabel('Year')
        plt.ylabel('Total Annual Costs (MM$)')
        plt.title('Total Annual Project Costs')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()
        if show:
            plt.show()


if __name__ == "__main__":
    pass    