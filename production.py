import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

class YearlyProductionProfile:
    """
    Yearly-based production simulator.
    - Type curve is generated daily (exponential) but aggregated to yearly bins.
    - Simulation runs year-by-year (fast) and applies peak_production as
      a yearly cap with proportional scaling across wells preserving EUR.
    """

    def __init__(self, production_duration: int):
        self.production_duration = production_duration  # years
        self.year_count = production_duration
        self.yearly_type_rate = None       # array: annual production per well by year (MMcf/year)
        self.yearly_type_rate_per_day = None  # optional if you need daily-equivalent
        self.yearly_type_cum = None        # cumulative per-well (MMcf)
        self.yearly_type_rate_interp = None
        self.yearly_drilling_plan = None

    def generate_type_curve_from_exponential(self, qi_mmcfd: float, EUR_target_mmcf: float, T_years: int):
        """
        Generate a high-resolution daily exponential type curve given qi (MMcf/d),
        decline D solved for target EUR (MMcf overall over T_years).
        Then aggregate into yearly production amounts (MMcf/year).
        """
        def eur_func(D):
            if D <= 0:
                return np.inf
            days = int(T_years * 365)
            t = np.linspace(0, T_years, days)
            q = qi_mmcfd * np.exp(-D * t)
            eur = np.sum(q)  # in MMcf-days but q is MMcf/day so sum over days -> MMcf
            return eur - EUR_target_mmcf

        # solve for D
        D_values = np.linspace(1e-6, 5.0, 2000)
        eur_vals = [eur_func(D) for D in D_values]
        sign_changes = np.where(np.diff(np.sign(eur_vals)))[0]
        if len(sign_changes) == 0:
            raise ValueError("Unable to bracket decline D. Check qi/EUR_target/T_years.")
        D_lower = D_values[sign_changes[0]]
        D_upper = D_values[sign_changes[0] + 1]
        res = root_scalar(eur_func, bracket=[D_lower, D_upper], method='brentq')
        D = res.root

        # daily type curve and then convert to annual bins
        days = int(T_years * 365)
        t = np.linspace(0, T_years, days)
        q_daily = qi_mmcfd * np.exp(-D * t)  # MMcf/day
        # annual production in MMcf/year: sum daily in each year
        yearly_rates = []
        for y in range(T_years):
            s = y * 365
            e = min((y + 1) * 365, len(q_daily))
            yearly_rates.append(np.sum(q_daily[s:e]))  # MMcf for that year
        yearly_rates = np.array(yearly_rates)  # length T_years, MMcf/year

        # store per-well yearly rate and cum
        self.D = D
        self.qi = qi_mmcfd
        self.T_years = T_years
        self.yearly_type_rate = yearly_rates  # MMcf per year for age 0..T_years-1
        self.yearly_type_cum = np.cumsum(self.yearly_type_rate)  # MMcf cumulative per well by year
        # interpolation: cum (MMcf) -> annual_rate (MMcf/year)
        self.yearly_type_rate_interp = lambda cum: np.interp(cum, self.yearly_type_cum, self.yearly_type_rate,
                                                            left=self.yearly_type_rate[0],
                                                            right=self.yearly_type_rate[-1])
        # self.plot_type_curve()

    def plot_type_curve(self):
        """Plot the generated type curve."""
        if self.yearly_type_rate is None:
            raise ValueError("Type curve has not been generated. Call generate_type_curve() first.")

        t_years = np.linspace(0, len(self.yearly_type_rate), len(self.yearly_type_rate))
        # plt.figure(figsize=(6, 4))
        fig,  (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(t_years, self.yearly_type_rate, label=f'Decline | D = {self.D:.3f} /yr')
        ax1.set_xlabel('Time (years)')
        ax1.set_ylabel('Gas Rate (MMcf/day)')
        ax1.set_title('Type Curve')
        ax1.grid(True)
        ax1.legend()
        # ax2 = ax1.twinx()
        ax2.plot(t_years, self.yearly_type_cum/1000, marker='o', color='black', label='Cumulative (Bcf)')
        ax2.set_ylabel('Cumulative Production (Bcf)')
        ax2.grid(True)
        ax2.legend(loc='best')
        plt.tight_layout()
        plt.show()

    def make_drilling_plan(self, total_wells_number: int , drilling_rate: int):
            """
            Create a yearly drilling plan based on the specified drilling rate.
            Parameters:
                total_wells_number: int - total number of wells to be drilled
                drilling_rate: float - number of wells to be drilled per year

            Returns:
                yearly_drilling_plan: dict - {year: wells_drilled}
            """
            # 시추 속도(drilling rate)를 감안하여 총 걸리는 기간을 계산
            years = int(total_wells_number // drilling_rate)
            remainder = total_wells_number % drilling_rate
            # 내년 시추되는 시추공수 계산
            yearly_drilling = [drilling_rate] * years
            if remainder > 0:
                yearly_drilling.append(remainder)

            # 1. Make years list
            drilling_years = [i for i in range(len(yearly_drilling))]

            # 2. Map year → drilling amount
            self.yearly_drilling_plan = dict(zip(drilling_years, yearly_drilling))

            return self.yearly_drilling_plan

    def make_production_profile_yearly(self, peak_production_annual=None):
        """
        Simulate year-by-year.
        - peak_production_annual: limit on field annual production (MMcf/year). If None -> no cap.
        Returns:
            yearly_field_production: dict {calendar_year: MMcf}
        """
        if self.yearly_type_rate is None or self.yearly_drilling_plan is None:
            raise ValueError("Type curve or drilling plan not set.")

        start = 0 #self.drill_start_year
        duration = self.production_duration
        field_prod = {}
        cumulative_field = 0.0

        # Each well is tracked by: {'cum': MMcf, 'age': years, 'start_year': int}
        wells = []

        for i in range(duration):
            year = start + i

            # add newly drilled wells at start of this year if any
            new_wells = int(self.yearly_drilling_plan.get(year, 0))
            for _ in range(new_wells):
                wells.append({'cum': 0.0, 'age': 0, 'start_year': year})

            # compute unconstrained annual production per well from type curve (based on age/cumulative)
            per_well_rates = []
            for w in wells:
                # rate according to well's cumulative position
                rate = self.yearly_type_rate_interp(w['cum'])
                per_well_rates.append(rate)

            total_unconstrained = sum(per_well_rates)

            # apply peak if needed (peak_production_annual is in MMcf/year)
            if (peak_production_annual is not None) and (total_unconstrained > peak_production_annual):
                scale = peak_production_annual / total_unconstrained
            else:
                scale = 1.0

            # update wells with scaled production for this year
            produced_this_year = 0.0
            for idx, w in enumerate(wells):
                produced = per_well_rates[idx] * scale
                w['cum'] += produced
                produced_this_year += produced
                w['age'] += 1

            field_prod[year] = produced_this_year / 1000 # to make BCF/year
            cumulative_field += produced_this_year

        self.yearly_field_production = field_prod
        self.cumulative_field = cumulative_field
        return field_prod

    def plot_yearly(self):
        if not hasattr(self, 'yearly_field_production'):
            raise ValueError("No yearly production available. Run make_production_profile_yearly().")
        years = sorted(self.yearly_field_production.keys())
        vals = [self.yearly_field_production[y] for y in years]
        cum = np.cumsum(vals)

        # fig, ax1 = plt.figure(figsize=(10,6))
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        ax1.bar(years, np.array(vals), label='Annual Production (Bcf)')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Annual Production (Bcf)')
        ax1.legend(loc='best')
        ax1.grid(True)
        ax1.set_title('Yearly Field Production')
        ax2 = ax1.twinx()
        ax2.plot(years, cum, marker='o', color='black', label='Cumulative (Bcf)')
        ax2.set_ylabel('Cumulative Production (Bcf)')
        ax2.legend(loc='best')




if __name__ == "__main__":
    # Example usage
    profile = YearlyProductionProfile(production_duration=20)
    profile.generate_type_curve_from_exponential(qi_mmcfd=10, EUR_target_mmcf=5000, T_years=20)
    profile.make_drilling_plan(total_wells_number=50, drilling_rate=5)
    profile.make_production_profile_yearly(peak_production_annual=300)
    profile.plot_yearly()