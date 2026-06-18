# Data Dictionary

The CSV is synthetic and simplified, but resembles policy-month life insurance data.

| Column | Meaning |
|---|---|
| policy_id | Unique policy identifier |
| snapshot_date | Monthly snapshot date |
| issue_date | Policy issue date |
| status | Current policy status, e.g. active/lapsed |
| product_type | Insurance product type |
| state | Policyholder state |
| urbanicity | Simplified urban/suburban/rural indicator |
| distribution_channel | Sales or servicing channel |
| payment_mode | Payment frequency/mode |
| autopay_flag | Whether autopay is enabled |
| current_age | Policyholder age at snapshot |
| income_estimate | Estimated annual household income |
| face_amount | Coverage amount |
| annualized_premium | Annualized premium amount |
| premium_paid_last_3m | Premium paid in the last three months |
| missed_payment_count_6m | Missed payment count in the last six months |
| agent_touch_count_12m | Agent/service interactions in the last twelve months |
| cash_value | Current cash value, if applicable |
| loan_balance | Policy loan balance, if applicable |
| address_change_12m | Whether address changed in the last twelve months |
| beneficiary_change_12m | Whether beneficiary changed in the last twelve months |
| lapse_next_month | Convenience training label. Candidates may use it directly or reconstruct a target, but should discuss leakage and timing assumptions. |
