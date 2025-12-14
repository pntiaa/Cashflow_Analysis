def main():
    print("Hello from cashflow-analysis!")


def economic_analysis(initial_param, dev_param, display_plots=False, output=False):

    # 1. Production Profile Generation------------------------------
    profile = YearlyProductionProfile(
        production_duration= initial_param['project_years'])

    ## type curve generation
    profile.generate_type_curve_from_exponential(
        qi_mmcfd=40.0,
        EUR_target_mmcf=60_000,
        T_years=30)  # 예시값

    ## drilling plan set up
    well_EUR_in_bcf = 60
    GIIP_in_bcf = initial_param['gas_reserves_in_bcf']
    OIIP_in_mmbbl = initial_param['oil_reserves_in_mmbbl']
    CGR =  OIIP_in_mmbbl / GIIP_in_bcf * 1000
    total_wells_number = math.ceil(initial_param['gas_reserves_in_bcf']/well_EUR_in_bcf) # target EUR 60 bcf

    yearly_drilling_plan = profile.make_drilling_plan(total_wells_number=total_wells_number, drilling_rate=12)

    gas_profile = profile.make_production_profile_yearly(peak_production_annual=initial_param['max_production_rate_in_mmcf'])  # MMcf/year
    oil_profile = {year: gas_prod * CGR / 1000 for year, gas_prod in gas_profile.items()} # in MMbbls

    # 2. Development Cost Generation------------------------------
    dev = DevelopmentCost(
        dev_start_year=2026,
        dev_param=dev_param,
        development_case='FPSO_case')

    dev.set_drilling_schedule(
        drill_start_year=initial_param['prod_well_drilling_start_year'],
        yearly_drilling_schedule = profile.yearly_drilling_plan,
        output=output)
    dev.set_annual_production(
        annual_gas_production=gas_profile,
        annual_oil_production=oil_profile,
        output=output)

    # calculating fixed opex
    fixed_opex_portion = 0.5
    opex_per_well = 251.15 # MM$
    opex_per_bcf = 2.093  # MM$ per BCF
    opex_fixed = total_wells_number * opex_per_well * fixed_opex_portion / dev._total_production_duration
    opex_variable = opex_per_bcf * (1- fixed_opex_portion )
    dev_param['FPSO_case']['OPEX_fixed'] = opex_fixed
    dev_param['FPSO_case']['OPEX_per_bcf'] = opex_variable

    # total cost
    summary = dev.calculate_total_costs(production_years=30, study_timing='year_0', facility_timing='year_1', output=output)

    # 3. Oil Price generation ------------------------------
    price = PriceDeck(
        start_year = initial_param['prod_well_drilling_start_year'],
        end_year = int(initial_param['prod_well_drilling_start_year']+40),
        oil_price_initial = initial_param['oil_price_per_barrel'],
        gas_price_initial =  initial_param['gas_price_per_mcf'],
        inflation_rate = initial_param['inflation_rate'],
        flat_after_year = None)

    # 4. Cash Flow Generation
    cf = CashFlow_KOR_Regime(
                            base_year = 2024,
                            oil_price_by_year = price.oil_price_by_year,
                            gas_price_by_year = price.gas_price_by_year,
                            cost_inflation_rate = initial_param['inflation_rate'],
                            discount_rate = initial_param['discount_rate'],
                            exchange_rate = initial_param['exchange_rate'])

    cf.set_development_costs(dev, output=output)
    cf.set_production_profile_from_dicts(
        oil_dict=dev.annual_oil_production,
        gas_dict=dev.annual_gas_production)

    cf.calculate_annual_revenue(output=output)
    cf.calculate_depreciation(method='straight_line', useful_life=10, output=output)
    cf.calculate_royalty()
    cf.calculate_taxes(output=output)
    cf.calculate_net_cash_flow(output=output)
    npv = cf.calculate_npv(output=output)
    # define output
    summary = cf.get_project_summary()
    if display_plots:
        # plot_cf_waterfall_chart(cf, width=600,height=400)
         plot_cash_flow_profile_plotly(cf, width=1200,height=800)
        # summary_plot(cf, width=1200,height=800)

    return summary


if __name__ == "__main__":
    main()
