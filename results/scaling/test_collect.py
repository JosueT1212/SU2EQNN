import tempfile
from pathlib import Path

from collect import collect

with tempfile.TemporaryDirectory() as d:
    (Path(d) / "top_tagging_bivector_patch_full_b12_10k_seed2.log").write_text(
        "COMMIT=x\n[..][__main__][INFO] - Test AUC=0.9712 | accuracy=0.9100\nEXIT_STATUS=0 at now\n")
    (Path(d) / "top_tagging_r1_full_b12_1M_seed0.log").write_text("COMMIT=x\nStep 5 | val AUC=0.9\n")
    rows, missing = collect(d)
    assert rows == [("bivector_patch", 10_000, 2, 0.9712)], rows
    assert missing == ["top_tagging_r1_full_b12_1M_seed0.log"], missing
print("ok")
