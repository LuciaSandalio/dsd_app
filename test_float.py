
try:
    val = float("0000.127")
    print(f"Parsed '0000.127' as: {val} (Type: {type(val)})")
except Exception as e:
    print(f"Error: {e}")

try:
    val2 = float("0000,127")
    print(f"Parsed '0000,127' as: {val2}")
except Exception as e:
    print(f"Error parsing comma: {e}")
