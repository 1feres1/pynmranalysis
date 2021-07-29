from pynmranalysis import nmrfunctions
import pandas as pd
def test_binning():
    """assert nmrfunctions.binning(52.370216, 4.895168, 52.520008,
    13.404954) == 945793.4375088713"""
    spec = pd.read_csv("x.csv")
    print(nmrfunctions.binning(spec , spec.columns , 0.04))