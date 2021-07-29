from pynmranalysis import normalization
import pandas as pd
def test_mean_normalization():
    """assert nmrfunctions.binning(52.370216, 4.895168, 52.520008,
    13.404954) == 945793.4375088713"""
    spec = pd.read_csv("x.csv")
    print(normalization.mean_normalization(spec,verbose=True))
test_mean_normalization()