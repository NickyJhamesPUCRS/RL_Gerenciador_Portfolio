# RL for portfolio management framework

## Install

- Create and environment, you could use conda.
- 'conda activate name_of_your_env'
- pip install -r requirements.txt
- you will need to change the timeseries.py from pyfolio lib:
+ change  'to_pydatetime' to: pd.to_datetime in ALL lines.
