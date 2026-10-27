"""Adaptive User Risk Profile: continuously updated model of a user's normal behaviour."""
import math
from collections import deque
import numpy as np


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371.0 * 2 * math.asin(min(1.0, math.sqrt(a)))


class UserProfile:
    def __init__(self, uid):
        self.uid = uid
        self.n = 0                      # trusted (learned-from) transactions
        self.mean, self.var = 6.0, 0.25  # log-amount statistics
        self.hist = np.full(24, 0.05)   # hour-of-day histogram (decaying)
        self.devices = set()
        self.lat = self.lon = None      # adaptive home centroid
        self.recent = deque(maxlen=50)  # timestamps (hours) of ALL attempts
        self.flag_ema = 0.0             # recent share of flagged transactions

    @property
    def std(self):
        return max(math.sqrt(self.var), 0.3)

    def record_attempt(self, ts):
        self.recent.append(ts)

    def note_outcome(self, flagged):
        self.flag_ema = 0.9 * self.flag_ema + 0.1 * float(flagged)

    def update(self, t):
        """Learn from a trusted transaction (allowed, or verified by the user)."""
        la = math.log(max(t["amount"], 1.0))
        self.n += 1
        a = max(1.0 / self.n, 0.05)
        if self.n == 1:
            self.mean, self.var = la, 0.25
            self.lat, self.lon = t["lat"], t["lon"]
        else:
            d = la - self.mean
            self.mean += a * d
            self.var = (1 - a) * (self.var + a * d * d)
            self.lat += a * (t["lat"] - self.lat)
            self.lon += a * (t["lon"] - self.lon)
        self.hist *= 0.99
        self.hist[int(t["hour"]) % 24] += 1.0
        self.devices.add(t["device_id"])


class ProfileStore(dict):
    def get_profile(self, uid):
        if uid not in self:
            self[uid] = UserProfile(uid)
        return self[uid]
