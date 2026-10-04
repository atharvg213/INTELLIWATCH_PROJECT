p = "scripts/validate_system_http_contracts.py"
s = open(p, encoding="utf-8").read()
bad = 'assert resp["deleted"] is True\\n    st2, _, b2 = get("/api/v1/cameras/cam_loading_bay_01/calibration")\\n    assert json.loads(b2.decode("utf-8"))["calibration_status"] == "UNCONFIGURED"'
good = ('assert resp["deleted"] is True\n'
        '    st2, _, b2 = get("/api/v1/cameras/cam_loading_bay_01/calibration")\n'
        '    assert json.loads(b2.decode("utf-8"))["calibration_status"] == "UNCONFIGURED"')
assert bad in s
open(p, "w", encoding="utf-8").write(s.replace(bad, good))
print("fixed")
