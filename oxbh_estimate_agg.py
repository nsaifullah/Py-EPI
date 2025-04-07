import pandas as pd
import numpy as np


def charge_estimate_agg(claims, min_n, base_agg_keys, agg_key_d):
    base_agg = claims.groupby(base_agg_keys, as_index=False).agg(base_n=('total_code_cost', 'count'),
                                                                 base_mean=('total_code_cost', 'mean'))
    meets_min_msk = base_agg['base_n'].ge(min_n)
    base_agg['status'] = 'not_defined'
    base_agg.loc[meets_min_msk, 'status'] = 'by_fac_code'
    for agg_level, agg_keys in agg_key_d.items():
        base_agg[agg_level + '_mean'] = base_agg.groupby(agg_keys)['base_mean'].transform('mean')  # sub weighted mean
        base_agg[agg_level + '_count'] = base_agg.groupby(agg_keys)['base_n'].transform('sum')
        base_agg.loc[~meets_min_msk & base_agg[agg_level + '_count'].ge(min_n), 'status'] = agg_level

    return base_agg


rng = np.random.default_rng(seed=123456)
n_clms = 100
n_facs = 12
n_codes = 20
estimable_threshold = 5

system_mapping_df = pd.DataFrame({'facility_npi': np.arange(0, n_facs),
                                  'hospital_system': ['System_A'] * 4 + ['System_B'] + ['System_C'] * 3
                                                     + ['System_D'] * 3 + ['System_E']})
code_bundle_mapping_df = pd.DataFrame({'code': np.arange(0, n_codes),
                                       'bundle': ['B_1'] * 8 + ['B_2'] * 3 + ['B_3'] * 3 + ['B_4'] * 4 + ['B_5'] * 2})
bundle_group_mapping_df = pd.DataFrame({'bundle': ['B_' + str(n+1) for n in range(0, 5)],
                                        'bundle_group': ['BG_1'] * 2 + ['BG_2'] + ['BG_3'] * 2})
facility_distro = [0.2, 0.1, 0.05, 0.05, 0.05, 0.025, 0.025, 0.2, 0.15, 0.05, 0.05, 0.05]
code_distro = [0.13] * 4 + [0.05] * 8 + [0.01] * 8

clm_id_list = ['0000_' + str(n) for n in np.arange(0, n_clms)]
fac_list = list(rng.choice(a=n_facs, size=n_clms, p=facility_distro, replace=True, shuffle=False))
code_list = list(rng.choice(a=n_codes, size=n_clms, p=code_distro, replace=True, shuffle=False))
clms_df = pd.DataFrame({'claim_id': clm_id_list, 'facility_npi': fac_list, 'main_code': code_list})

clms_df = pd.merge(clms_df, system_mapping_df, how='left', on='facility_npi', validate='m:1')
clms_df = pd.merge(clms_df, code_bundle_mapping_df, how='left', left_on='main_code', right_on='code', validate='m:1')
clms_df = pd.merge(clms_df, bundle_group_mapping_df, how='left', on='bundle', validate='m:1')
for code in range(0, n_codes):
    clms_df.loc[clms_df['main_code'].eq(code), 'total_code_cost'] = rng.lognormal(mean=1*code, sigma=0.9**code)
clms_df.sort_values(['facility_npi', 'main_code'], inplace=True)

extra_agg_d = {'by_fac_bundle': ['facility_npi', 'bundle'], 'by_fac_bundle_group': ['facility_npi', 'bundle_group']}

results_df = charge_estimate_agg(clms_df, estimable_threshold, ['facility_npi', 'main_code', 'bundle', 'bundle_group']
                                 , extra_agg_d)
print(results_df['status'].value_counts(normalize=True))
