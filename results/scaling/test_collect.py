import tempfile
from pathlib import Path

from collect import collect, collect_full

with tempfile.TemporaryDirectory() as d:
    (Path(d) / "top_tagging_bivector_patch_full_b12_10k_seed2.log").write_text(
        "COMMIT=x\n[..][__main__][INFO] - Test AUC=0.9712 | accuracy=0.9100\nEXIT_STATUS=0 at now\n")
    (Path(d) / "top_tagging_r1_full_b12_1M_seed0.log").write_text("COMMIT=x\nStep 5 | val AUC=0.9\n")
    (Path(d) / "top_tagging_rest_durham_plane_normal_full_b12_full_seed0.log").write_text(
        "COMMIT=x JOB=1_0 HOST=h GPU=0 N=1211000 STEPS=1 VAL_EVERY=1\nTest AUC=0.9850 | a\nEXIT_STATUS=0 at now\n")
    (Path(d) / "top_tagging_rest_xyplane_full_b12_30k_seed1.log").write_text(
        "COMMIT=x N=30000\nTest AUC=0.9700 | a\nEXIT_STATUS=0 at now\n")
    (Path(d) / "top_tagging_rest_durham_masses_patch_full_b12_30k_seed0.log").write_text(
        "COMMIT=x N=30000\nTest AUC=0.9750 | a\nEXIT_STATUS=0 at now\n")
    (Path(d) / "top_tagging_subjet_mass_patch_xyplane_full_b12_30k_seed0.log").write_text(
        "COMMIT=x N=30000\nTest AUC=0.9760 | a\nEXIT_STATUS=0 at now\n")
    rows, missing = collect(d)
    assert sorted(rows) == sorted([
        ("subjet_mass_patch_little", 30_000, 0, 0.9750),
        ("subjet_mass_patch", 30_000, 0, 0.9760),
        ("bivector_patch", 10_000, 2, 0.9712),
        ("pseudoscalar_triad_little", 1_211_000, 0, 0.9850),
        ("r1_little", 30_000, 1, 0.9700),
    ]), rows
    assert missing == ["top_tagging_r1_full_b12_1M_seed0.log"], missing
with tempfile.TemporaryDirectory() as d:
    (Path(d) / "seed_0").mkdir()
    (Path(d) / "seed_0" / "top_tagging_rest_durham_patch_full_b12.log").write_text(
        "COMMIT=x JOB=1_0 HOST=h GPU=0 N=1211000 STEPS=1 TAG=full START=now\nTest AUC=0.9858 | a\nEXIT_STATUS=0 at now\n")
    (Path(d) / "seed_0" / "top_tagging_r1_full_b12.log").write_text(   # old sweep log: ignored
        "COMMIT=x JOB=3475805_0 HOST=h\nTest AUC=0.9850 | a\nEXIT_STATUS=0 at now\n")
    (Path(d) / "seed_0" / "top_tagging_grade2_bivectors_xyplane_full_b12.log").write_text(
        "COMMIT=x N=1211000 TAG=full\nStep 5000 | val AUC=0.98\n")
    (Path(d) / "seed_0" / "top_tagging_subjet_mass_full_b12.log").write_text(   # naive +-e3 full: not official
        "COMMIT=x N=1211000 TAG=full\nTest AUC=0.9840 | a\nEXIT_STATUS=0 at now\n")
    (Path(d) / "seed_0" / "top_tagging_subjet_mass_patch_xyplane_full_b12.log").write_text(
        "COMMIT=x N=1211000 TAG=full\nTest AUC=0.9855 | a\nEXIT_STATUS=0 at now\n")
    rows, missing = collect_full(d)
    assert sorted(rows) == sorted([("bivector_patch_little", 1_211_000, 0, 0.9858),
                                   ("subjet_mass_patch", 1_211_000, 0, 0.9855)]), rows
    assert missing == ["seed_0/top_tagging_grade2_bivectors_xyplane_full_b12.log"], missing
print("ok")
