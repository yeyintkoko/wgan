#### How to Run

1. Create a virtual env and activate it.

```
source ./venv/bin/active
```

2. Install dependencies.

```
pip install -r requirements.txt
```

3. Run
   Run WGAN implementation for timeseries prediction. Here in gan.py, the apple stock data for last 10 years are analysed and trained with 80% historical data and tested with 20% historical data. Predicted for next 30 days.

```
python3 gan.py
```

Wait around 5 to 30 min while it is training, it will keep showing ui at closer results.
