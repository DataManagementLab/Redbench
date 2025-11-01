def toggle_suffix(s):
    if s.endswith("_0"):
        return s[:-2] + "_1"
    elif s.endswith("_1"):
        return s[:-2] + "_0"
    return s
