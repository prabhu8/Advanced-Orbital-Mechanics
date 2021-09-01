import pandas as pd
import numpy as np

Solar_S = pd.DataFrame(index=["Units", "Sun", "Moon", "Earth"],
                       columns=["a", "miu", "r", "Per", "e", "i"])
Solar_S.loc["Units"] = ["km", "km^3/s^2", "km", "s", "-", "deg"]
Solar_S.loc["Sun"] = [np.NAN, 13271244017.990, 695990.00, np.NaN, np.NAN,
                      np.NAN]
Solar_S.loc["Moon"] = [384400.000, 4902.8011, 1738.0, 2360592, 0.0549, 5.145]
Solar_S.loc["Earth"] = [149597870.7, 398600.4415, 6378.136, 31558148.63,
                        0.01671022, 0.00005]

G = 6.67408 * 10**-20
