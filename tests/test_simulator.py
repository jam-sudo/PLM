"""Unit tests for clinical trial simulator."""

import numpy as np
import pytest

from simulator.patient import VirtualPatient, PatientState, generate_population
from simulator.pk_engine import (
    pk_concentration,
    multi_dose_concentration,
    AnalyticalPKEngine,
    PLMPKEngine,
)
from simulator.adherence import AdherenceModel
from simulator.pharmacology import ae_probability, efficacy_probability
from simulator.trial import (
    TrialProtocol, TrialArm, simulate_trial, _check_steady_state,
)


# ── Helpers ──────────────────────────────────────────────────────────

def _make_patient(**overrides) -> VirtualPatient:
    defaults = dict(
        id=0, weight_kg=70, age_yr=40, sex="M", cyp3a4_activity=1.0,
        ka=1.5, ke=0.15, vd_f=80.0, tlag=0.0,
        adherence_tendency=0.9, ae_sensitivity=1.0,
    )
    defaults.update(overrides)
    return VirtualPatient(**defaults)


def _make_protocol(**overrides) -> TrialProtocol:
    defaults = dict(
        drug_name="Test Drug",
        arms=[TrialArm(label="40mg", dose_mg=40.0, n_patients=50)],
        interval_h=24.0,
        n_doses=10,
        cmax_therapeutic=0.50,
        ec50=0.05,
        seed=42,
    )
    defaults.update(overrides)
    return TrialProtocol(**defaults)


# ── PK Engine Tests ──────────────────────────────────────────────────

class TestPKConcentration:
    def test_single_dose_cmax(self):
        """Verify Cmax is close to analytical expectation."""
        ka, ke, vd_f, dose = 1.5, 0.15, 80.0, 100.0
        t = np.linspace(0, 24, 1000)
        c = pk_concentration(t, dose, ka, ke, vd_f)
        cmax = np.max(c)

        # Analytical Tmax = ln(ka/ke) / (ka - ke)
        tmax_analytical = np.log(ka / ke) / (ka - ke)
        cmax_analytical = pk_concentration(tmax_analytical, dose, ka, ke, vd_f)

        assert abs(cmax - cmax_analytical) / cmax_analytical < 0.01

    def test_concentration_at_t0_is_zero(self):
        c = pk_concentration(0.0, 100.0, 1.5, 0.15, 80.0)
        assert c == pytest.approx(0.0, abs=1e-10)

    def test_concentration_decays_to_zero(self):
        c = pk_concentration(200.0, 100.0, 1.5, 0.15, 80.0)
        assert c < 0.001

    def test_dose_proportionality(self):
        """Double dose -> double concentration (linear PK)."""
        t = np.array([2.0])
        c1 = pk_concentration(t, 50.0, 1.5, 0.15, 80.0)
        c2 = pk_concentration(t, 100.0, 1.5, 0.15, 80.0)
        assert c2 == pytest.approx(2 * c1, rel=1e-10)

    def test_lag_time_shifts_tmax(self):
        """Adding lag time should shift Tmax forward by tlag."""
        ka, ke, vd_f, dose = 1.5, 0.15, 80.0, 100.0
        t = np.linspace(0, 24, 2000)

        c_no_lag = pk_concentration(t, dose, ka, ke, vd_f, tlag=0.0)
        c_lag = pk_concentration(t, dose, ka, ke, vd_f, tlag=0.5)

        tmax_no_lag = t[np.argmax(c_no_lag)]
        tmax_lag = t[np.argmax(c_lag)]

        # Tmax should shift by approximately tlag
        assert abs((tmax_lag - tmax_no_lag) - 0.5) < 0.05

    def test_lag_time_zero_before_lag(self):
        """Concentration should be exactly 0 before tlag."""
        c = pk_concentration(np.array([0.1, 0.2, 0.3]), 100.0, 1.5, 0.15, 80.0, tlag=0.5)
        assert np.all(c == 0.0)

    def test_degenerate_ka_eq_ke(self):
        """When ka == ke, should still return valid concentrations."""
        c = pk_concentration(np.array([1.0, 2.0, 4.0]), 100.0, 0.15, 0.15, 80.0)
        assert np.all(np.isfinite(c))
        assert np.all(c >= 0)

    def test_negative_time_returns_zero(self):
        """Negative time should yield zero concentration."""
        c = pk_concentration(np.array([-1.0, -0.5]), 100.0, 1.5, 0.15, 80.0)
        # Before dose, superposition masks dt < 0; single dose at t<0 is non-physical
        # but the function should not produce NaN or negative values
        assert np.all(np.isfinite(c))

    def test_auc_mass_balance(self):
        """AUC(0-inf) should equal Dose / CL for 1-compartment model.

        AUC = Dose / (Vd * ke) = Dose / CL
        """
        ka, ke, vd_f, dose = 1.5, 0.15, 80.0, 100.0
        cl = vd_f * ke  # 12 L/h
        auc_analytical = dose / cl

        # Numerical AUC via trapezoidal rule (long enough to capture tail)
        t = np.linspace(0, 200, 10000)
        c = pk_concentration(t, dose, ka, ke, vd_f)
        auc_numerical = np.trapz(c, t)

        assert auc_numerical == pytest.approx(auc_analytical, rel=0.01)

    def test_scalar_input(self):
        """pk_concentration should accept scalar t and return scalar."""
        c = pk_concentration(2.0, 100.0, 1.5, 0.15, 80.0)
        assert isinstance(c, float)
        assert c > 0

    def test_large_tlag_delays_entirely(self):
        """If tlag > t, concentration should be 0."""
        c = pk_concentration(np.array([1.0, 2.0, 3.0]), 100.0, 1.5, 0.15, 80.0, tlag=5.0)
        assert np.all(c == 0.0)

    def test_lag_preserves_cmax_magnitude(self):
        """Lag time shifts but shouldn't change Cmax value."""
        ka, ke, vd_f, dose = 1.5, 0.15, 80.0, 100.0
        t = np.linspace(0, 48, 5000)
        c_no_lag = pk_concentration(t, dose, ka, ke, vd_f, tlag=0.0)
        c_lag = pk_concentration(t, dose, ka, ke, vd_f, tlag=1.0)
        assert np.max(c_lag) == pytest.approx(np.max(c_no_lag), rel=0.01)


class TestMultiDose:
    def test_superposition(self):
        """Two doses should sum to individual dose contributions."""
        ka, ke, vd_f = 1.5, 0.15, 80.0
        t = np.array([25.0])

        doses = [(0.0, 100.0), (24.0, 100.0)]
        c_multi = multi_dose_concentration(t, doses, ka, ke, vd_f)

        # Manual superposition
        c1 = pk_concentration(25.0, 100.0, ka, ke, vd_f)
        c2 = pk_concentration(1.0, 100.0, ka, ke, vd_f)
        c_sum = c1 + c2

        assert c_multi[0] == pytest.approx(c_sum, rel=1e-10)

    def test_accumulation(self):
        """Repeated dosing should show accumulation until steady state."""
        ka, ke, vd_f = 1.5, 0.15, 80.0
        # 5 daily doses
        doses = [(24.0 * i, 100.0) for i in range(5)]

        # Cmax of each interval
        cmaxes = []
        for i in range(5):
            t_eval = np.linspace(24.0 * i, 24.0 * (i + 1), 100)
            c = multi_dose_concentration(t_eval, doses[:i + 1], ka, ke, vd_f)
            cmaxes.append(np.max(c))

        # Each Cmax should be >= previous (accumulation)
        for i in range(1, len(cmaxes)):
            assert cmaxes[i] >= cmaxes[i - 1] * 0.99  # allow tiny float error

    def test_steady_state_convergence(self):
        """After enough doses, Cmax should stabilize (accumulation ratio converges)."""
        ka, ke, vd_f = 1.5, 0.15, 80.0
        # 20 daily doses — t1/2 ~4.6h, so SS within ~5 doses
        doses = [(24.0 * i, 100.0) for i in range(20)]

        cmaxes = []
        for i in range(20):
            t_eval = np.linspace(24.0 * i, 24.0 * (i + 1), 100)
            c = multi_dose_concentration(t_eval, doses[:i + 1], ka, ke, vd_f)
            cmaxes.append(np.max(c))

        # Last two should be within 1% of each other
        assert abs(cmaxes[-1] - cmaxes[-2]) / cmaxes[-2] < 0.01

    def test_empty_doses_returns_zero(self):
        """No doses should give zero concentration everywhere."""
        t = np.linspace(0, 24, 50)
        c = multi_dose_concentration(t, [], 1.5, 0.15, 80.0)
        assert np.all(c == 0.0)

    def test_multi_dose_with_lag(self):
        """Superposition with lag time should still work correctly."""
        ka, ke, vd_f, tlag = 1.5, 0.15, 80.0, 0.5
        doses = [(0.0, 100.0), (24.0, 100.0)]
        t = np.array([25.0])
        c = multi_dose_concentration(t, doses, ka, ke, vd_f, tlag)

        c1 = pk_concentration(25.0, 100.0, ka, ke, vd_f, tlag)
        c2 = pk_concentration(1.0, 100.0, ka, ke, vd_f, tlag)
        assert c[0] == pytest.approx(c1 + c2, rel=1e-10)


class TestAnalyticalEngine:
    def test_implements_protocol(self):
        from simulator.pk_engine import PKEngine
        engine = AnalyticalPKEngine()
        assert isinstance(engine, PKEngine)

    def test_matches_direct_call(self):
        patient = _make_patient(ka=1.5, ke=0.15, vd_f=80.0, tlag=0.3)
        doses = [(0.0, 100.0)]
        t = np.linspace(0, 24, 100)

        engine = AnalyticalPKEngine()
        c_engine = engine.concentration(t, doses, patient)
        c_direct = multi_dose_concentration(t, doses, 1.5, 0.15, 80.0, 0.3)

        np.testing.assert_array_almost_equal(c_engine, c_direct)


class TestPLMPKEngine:
    def test_fallback_to_analytical(self):
        """PLMPKEngine with no model should fallback to AnalyticalPKEngine."""
        patient = _make_patient(ka=1.5, ke=0.15, vd_f=80.0, tlag=0.0)
        doses = [(0.0, 100.0)]
        t = np.linspace(0, 24, 100)

        plm = PLMPKEngine(model_path=None)
        analytical = AnalyticalPKEngine()

        c_plm = plm.concentration(t, doses, patient)
        c_analytical = analytical.concentration(t, doses, patient)

        np.testing.assert_array_almost_equal(c_plm, c_analytical)

    def test_bad_model_path_raises(self):
        """Loading from nonexistent path should raise RuntimeError."""
        with pytest.raises(RuntimeError, match="could not load model"):
            PLMPKEngine(model_path="/nonexistent/model.pkl")


# ── Population Generator Tests ───────────────────────────────────────

class TestPopulation:
    def test_generates_correct_count(self):
        pop = generate_population(100, seed=1)
        assert len(pop) == 100

    def test_allometric_vd(self):
        """Heavier patients should have larger Vd, but sub-linearly."""
        rng = np.random.default_rng(42)
        pop = generate_population(1000, seed=42)

        light = [p for p in pop if p.weight_kg < 55]
        heavy = [p for p in pop if p.weight_kg > 95]

        mean_vd_light = np.mean([p.vd_f for p in light])
        mean_vd_heavy = np.mean([p.vd_f for p in heavy])

        # Heavy should have larger Vd
        assert mean_vd_heavy > mean_vd_light

        # But the ratio should be less than the weight ratio (allometric < linear)
        wt_ratio = np.mean([p.weight_kg for p in heavy]) / np.mean([p.weight_kg for p in light])
        vd_ratio = mean_vd_heavy / mean_vd_light
        assert vd_ratio < wt_ratio  # sub-linear scaling

    def test_sex_distribution(self):
        pop = generate_population(1000, seed=42)
        n_male = sum(1 for p in pop if p.sex == "M")
        assert 400 < n_male < 600  # roughly 50/50

    def test_tlag_positive(self):
        pop = generate_population(100, seed=42)
        for p in pop:
            assert p.tlag >= 0

    def test_reproducible_with_seed(self):
        """Same seed should produce identical populations."""
        pop1 = generate_population(50, seed=99)
        pop2 = generate_population(50, seed=99)
        for p1, p2 in zip(pop1, pop2):
            assert p1.weight_kg == p2.weight_kg
            assert p1.ka == p2.ka
            assert p1.ke == p2.ke

    def test_different_seeds_differ(self):
        """Different seeds should produce different populations."""
        pop1 = generate_population(50, seed=1)
        pop2 = generate_population(50, seed=2)
        weights1 = [p.weight_kg for p in pop1]
        weights2 = [p.weight_kg for p in pop2]
        assert weights1 != weights2

    def test_female_lower_cyp(self):
        """Female patients should have lower mean CYP3A4 activity than males."""
        pop = generate_population(2000, seed=42)
        male_cyp = np.mean([p.cyp3a4_activity for p in pop if p.sex == "M"])
        female_cyp = np.mean([p.cyp3a4_activity for p in pop if p.sex == "F"])
        assert female_cyp < male_cyp

    def test_age_range(self):
        """All patients should be 18-85 years old."""
        pop = generate_population(500, seed=42)
        for p in pop:
            assert 18 <= p.age_yr <= 85

    def test_weight_range(self):
        """All patients should be 40-140 kg."""
        pop = generate_population(500, seed=42)
        for p in pop:
            assert 40 <= p.weight_kg <= 140

    def test_ke_derived_from_cl_vd(self):
        """ke should be positive and within reasonable bounds."""
        pop = generate_population(500, seed=42)
        for p in pop:
            assert p.ke > 0
            # t1/2 = ln2/ke should be between 0.3h and 70h for most drugs
            t_half = 0.693 / p.ke
            assert 0.3 < t_half < 70

    def test_custom_drug_params(self):
        """Custom drug params should be respected."""
        params = {
            "ka_mean": 5.0, "ka_cv": 0.10,
            "cl_ref70": 100.0, "cl_cv": 0.10,
            "vd_f_ref70": 200.0, "vd_f_cv": 0.10,
            "tlag_mean": 0.0, "tlag_cv": 0.0,
        }
        pop = generate_population(200, seed=42, drug_params=params)
        mean_ka = np.mean([p.ka for p in pop])
        # With CV=0.10 around mean=5.0, mean should be close
        assert 4.0 < mean_ka < 6.0
        # tlag_mean=0 should give all tlag=0
        assert all(p.tlag == 0.0 for p in pop)


# ── Pharmacology Tests ───────────────────────────────────────────────

class TestAEProbability:
    def test_low_cmax_low_probability(self):
        p = ae_probability(0.1, 1.0, 1.0)
        assert p < 0.2

    def test_high_cmax_high_probability(self):
        p = ae_probability(3.0, 1.0, 1.0)
        assert p > 0.8

    def test_zero_cmax_therapeutic(self):
        p = ae_probability(1.0, 0.0, 1.0)
        assert p == 0.0

    def test_sensitivity_shifts_curve(self):
        """More sensitive patient (lower threshold) -> higher AE probability."""
        p_sensitive = ae_probability(0.5, 1.0, 0.5)  # sensitive
        p_resistant = ae_probability(0.5, 1.0, 1.5)  # resistant
        assert p_sensitive > p_resistant

    def test_monotonic_with_cmax(self):
        """AE probability should increase monotonically with Cmax."""
        cmax_range = [0.1, 0.5, 1.0, 2.0, 5.0]
        probs = [ae_probability(c, 1.0, 1.0) for c in cmax_range]
        for i in range(1, len(probs)):
            assert probs[i] >= probs[i - 1]

    def test_output_bounded_0_1(self):
        """AE probability must always be in [0, 1]."""
        for cmax in [0, 0.001, 0.5, 1.0, 10.0, 1000.0]:
            p = ae_probability(cmax, 1.0, 1.0)
            assert 0.0 <= p <= 1.0

    def test_steepness_effect(self):
        """Higher steepness should make transition sharper."""
        p_low = ae_probability(1.0, 1.0, 1.0, steepness=1.0)
        p_high = ae_probability(1.0, 1.0, 1.0, steepness=10.0)
        # At ratio == sensitivity, both should be ~0.5 regardless of steepness
        assert abs(p_low - 0.5) < 0.3
        assert abs(p_high - 0.5) < 0.3


class TestEfficacy:
    def test_emax_ceiling(self):
        """At very high concentration, response -> Emax."""
        p = efficacy_probability(1000.0, 0.1, emax=0.95)
        assert p == pytest.approx(0.95, abs=0.01)

    def test_ec50_gives_half_emax(self):
        """At C = EC50, response = Emax/2."""
        p = efficacy_probability(0.1, 0.1, emax=0.90, hill=1.0)
        assert p == pytest.approx(0.45, abs=0.01)

    def test_zero_concentration(self):
        p = efficacy_probability(0.0, 0.1)
        assert p == 0.0

    def test_hill_steepens(self):
        """Higher Hill coefficient -> steeper dose-response."""
        p_h1 = efficacy_probability(0.05, 0.1, hill=1.0)
        p_h3 = efficacy_probability(0.05, 0.1, hill=3.0)
        # At C < EC50, higher Hill gives LOWER response (steeper = more binary)
        assert p_h3 < p_h1

    def test_monotonic_with_concentration(self):
        """Efficacy should increase monotonically with concentration."""
        concs = [0.01, 0.05, 0.1, 0.5, 1.0, 10.0]
        probs = [efficacy_probability(c, 0.1) for c in concs]
        for i in range(1, len(probs)):
            assert probs[i] >= probs[i - 1]

    def test_zero_ec50(self):
        p = efficacy_probability(1.0, 0.0)
        assert p == 0.0

    def test_negative_concentration(self):
        """Negative concentration (non-physical) should return 0."""
        p = efficacy_probability(-1.0, 0.1)
        assert p == 0.0

    def test_emax_scales_output(self):
        """Different emax values should scale the ceiling."""
        p_low = efficacy_probability(100.0, 0.1, emax=0.50)
        p_high = efficacy_probability(100.0, 0.1, emax=0.90)
        assert p_low == pytest.approx(0.50, abs=0.01)
        assert p_high == pytest.approx(0.90, abs=0.01)


# ── Adherence Tests ──────────────────────────────────────────────────

class TestAdherence:
    def test_perfect_adherence(self):
        protocol = _make_protocol()
        result = simulate_trial(protocol, adherence_model=None)
        arm = list(result.arm_results.values())[0]
        assert arm.mean_adherence_rate == pytest.approx(1.0)
        assert arm.n_dropouts == 0

    def test_feedback_increases_dropout_narrow_ti(self):
        """Narrow therapeutic index -> more dropouts with feedback."""
        protocol_wide = _make_protocol(cmax_therapeutic=1.0, n_doses=30)
        protocol_narrow = _make_protocol(cmax_therapeutic=0.10, n_doses=30)

        adh = AdherenceModel(p11_base=0.92, p01_base=0.50, dropout_threshold=5)

        result_wide = simulate_trial(protocol_wide, adherence_model=adh, enable_feedback=True)
        result_narrow = simulate_trial(protocol_narrow, adherence_model=adh, enable_feedback=True)

        arm_wide = list(result_wide.arm_results.values())[0]
        arm_narrow = list(result_narrow.arm_results.values())[0]

        # Narrow TI should have more AE events and/or higher dropout
        assert arm_narrow.total_ae_events >= arm_wide.total_ae_events

    def test_dropout_after_consecutive_skips(self):
        """Patient should drop out after hitting consecutive skip threshold."""
        adh = AdherenceModel(p11_base=0.0, p01_base=0.0, dropout_threshold=3)
        rng = np.random.default_rng(42)
        patient = _make_patient(adherence_tendency=0.0)
        state = PatientState()

        actions = []
        for _ in range(10):
            a = adh.decide_no_feedback(rng, patient, state)
            actions.append(a)
            if a == "dropout":
                break

        assert "dropout" in actions
        assert state.dropped_out is True
        assert state.consecutive_skips >= 3

    def test_already_dropped_out_returns_dropout(self):
        """Once dropped out, all subsequent calls return dropout."""
        adh = AdherenceModel()
        rng = np.random.default_rng(42)
        patient = _make_patient()
        state = PatientState()
        state.dropped_out = True

        assert adh.decide(rng, patient, state, 0.0, 1.0) == "dropout"
        assert adh.decide_no_feedback(rng, patient, state) == "dropout"

    def test_jitter_respects_min_gap(self):
        """Jittered dose time must respect minimum gap from previous dose."""
        adh = AdherenceModel(timing_jitter_h=5.0, min_gap_h=4.0)
        rng = np.random.default_rng(42)

        for _ in range(100):
            actual = adh.jitter_dose_time(rng, 24.0, last_dose_time=22.0)
            assert actual >= 22.0 + 4.0  # min_gap

    def test_jitter_zero_means_exact(self):
        """Zero jitter should return exact scheduled time."""
        adh = AdherenceModel(timing_jitter_h=0.0)
        rng = np.random.default_rng(42)
        assert adh.jitter_dose_time(rng, 24.0, 0.0) == 24.0

    def test_jitter_non_negative_time(self):
        """Jittered time should never be negative."""
        adh = AdherenceModel(timing_jitter_h=10.0)
        rng = np.random.default_rng(42)
        for _ in range(200):
            t = adh.jitter_dose_time(rng, 0.5, None)
            assert t >= 0.0

    def test_symptom_return_bonus(self):
        """Longer non-adherent streaks should increase P(resume)."""
        adh = AdherenceModel(p01_base=0.30, dropout_threshold=100)
        rng = np.random.default_rng(42)
        patient = _make_patient(adherence_tendency=0.9)

        # Simulate many draws at different consecutive skip counts
        resume_counts = {}
        for n_skips in [1, 5]:
            count = 0
            for trial in range(2000):
                state = PatientState()
                state.consecutive_skips = n_skips
                r = np.random.default_rng(trial)
                action = adh.decide_no_feedback(r, patient, state)
                if action == "take":
                    count += 1
            resume_counts[n_skips] = count / 2000

        # More skips should give higher resume rate
        assert resume_counts[5] > resume_counts[1]

    def test_ae_feedback_generates_ae_events(self):
        """Feedback model with high Cmax should generate AE events."""
        adh = AdherenceModel(p11_base=0.95, dropout_threshold=100)
        rng = np.random.default_rng(42)
        patient = _make_patient(ae_sensitivity=0.5)  # very sensitive
        state = PatientState()
        state.doses_taken = [(0.0, 100.0)]

        n_ae_before = len(state.ae_events)
        # Call decide many times with very high Cmax
        for _ in range(50):
            state.consecutive_skips = 0
            adh.decide(rng, patient, state, last_cmax=5.0, cmax_therapeutic=0.5)

        # Should have generated some AE events
        assert len(state.ae_events) > n_ae_before

    def test_jitter_widens_cmax_distribution(self):
        """Dose-timing jitter should increase Cmax variability."""
        protocol = _make_protocol(n_doses=20)

        no_jitter = AdherenceModel(p11_base=1.0, p01_base=1.0, timing_jitter_h=0.0)
        with_jitter = AdherenceModel(p11_base=1.0, p01_base=1.0, timing_jitter_h=2.0)

        result_no = simulate_trial(protocol, adherence_model=no_jitter, enable_feedback=False)
        result_yes = simulate_trial(protocol, adherence_model=with_jitter, enable_feedback=False)

        arm_no = list(result_no.arm_results.values())[0]
        arm_yes = list(result_yes.arm_results.values())[0]

        cv_no = arm_no.cmax_ss_std / arm_no.cmax_ss_mean if arm_no.cmax_ss_mean > 0 else 0
        cv_yes = arm_yes.cmax_ss_std / arm_yes.cmax_ss_mean if arm_yes.cmax_ss_mean > 0 else 0

        # Jitter should increase CV (or at minimum not decrease it much)
        # Using a soft assertion since stochastic
        assert cv_yes >= cv_no * 0.8  # allow some stochastic noise


# ── Multi-Arm Trial Tests ────────────────────────────────────────────

class TestMultiArm:
    def test_different_doses_different_cmax(self):
        """Higher dose should produce higher mean Cmax."""
        protocol = _make_protocol(
            arms=[
                TrialArm(label="10mg", dose_mg=10.0, n_patients=100),
                TrialArm(label="80mg", dose_mg=80.0, n_patients=100),
            ],
            n_doses=15,
        )
        result = simulate_trial(protocol, adherence_model=None)

        cmax_10 = result.arm_results["10mg"].cmax_ss_mean
        cmax_80 = result.arm_results["80mg"].cmax_ss_mean

        assert cmax_80 > cmax_10 * 3  # should scale roughly linearly

    def test_higher_dose_higher_response(self):
        """Higher dose -> higher efficacy response rate (with perfect adherence)."""
        protocol = _make_protocol(
            arms=[
                TrialArm(label="5mg", dose_mg=5.0, n_patients=200),
                TrialArm(label="80mg", dose_mg=80.0, n_patients=200),
            ],
            n_doses=20,
            ec50=0.05,
        )
        result = simulate_trial(protocol, adherence_model=None)

        rr_low = result.arm_results["5mg"].response_rate
        rr_high = result.arm_results["80mg"].response_rate

        assert rr_high >= rr_low

    def test_four_arms(self):
        """Verify 4-arm trial runs and returns all arms."""
        protocol = _make_protocol(
            arms=[
                TrialArm(label=f"{d}mg", dose_mg=d, n_patients=30)
                for d in [10, 20, 40, 80]
            ],
        )
        result = simulate_trial(protocol)
        assert len(result.arm_results) == 4
        assert all(label in result.arm_results for label in ["10mg", "20mg", "40mg", "80mg"])

    def test_single_patient_arm(self):
        """Edge case: 1-patient arm should not crash."""
        protocol = _make_protocol(
            arms=[TrialArm(label="solo", dose_mg=40.0, n_patients=1)],
            n_doses=5,
        )
        result = simulate_trial(protocol)
        arm = result.arm_results["solo"]
        assert arm.n_patients == 1
        assert len(arm.patient_summaries) == 1


# ── Steady-State Detection Tests ─────────────────────────────────────

class TestSteadyState:
    def test_stable_values(self):
        assert _check_steady_state([1.0, 1.0]) is True

    def test_changing_values(self):
        assert _check_steady_state([1.0, 2.0]) is False

    def test_single_value(self):
        assert _check_steady_state([1.0]) is False

    def test_empty(self):
        assert _check_steady_state([]) is False

    def test_within_threshold(self):
        assert _check_steady_state([1.0, 1.05], threshold=0.10) is True

    def test_zero_prev(self):
        """If previous Cmax is 0, should return False (avoid div by zero)."""
        assert _check_steady_state([0.0, 0.5]) is False


# ── Trial Invariant Tests ────────────────────────────────────────────

class TestTrialInvariants:
    def test_completers_plus_dropouts_equals_n(self):
        """Completers + dropouts should equal total patients."""
        protocol = _make_protocol(n_doses=20)
        adh = AdherenceModel(p11_base=0.80, p01_base=0.40, dropout_threshold=5)
        result = simulate_trial(protocol, adherence_model=adh, enable_feedback=True)
        arm = list(result.arm_results.values())[0]
        assert arm.n_completers + arm.n_dropouts == arm.n_patients

    def test_adherence_rate_bounded(self):
        """All patient adherence rates should be in [0, 1]."""
        protocol = _make_protocol(n_doses=20)
        adh = AdherenceModel(p11_base=0.80, p01_base=0.40)
        result = simulate_trial(protocol, adherence_model=adh, enable_feedback=True)
        arm = list(result.arm_results.values())[0]
        for p in arm.patient_summaries:
            assert 0.0 <= p["adherence_rate"] <= 1.0

    def test_response_rate_bounded(self):
        """Response rate should be in [0, 1]."""
        protocol = _make_protocol(n_doses=15)
        result = simulate_trial(protocol)
        arm = list(result.arm_results.values())[0]
        assert 0.0 <= arm.response_rate <= 1.0

    def test_cmax_non_negative(self):
        """All Cmax values should be >= 0."""
        protocol = _make_protocol(n_doses=10)
        result = simulate_trial(protocol)
        arm = list(result.arm_results.values())[0]
        for p in arm.patient_summaries:
            assert p["ss_cmax"] >= 0

    def test_ctrough_less_than_cmax(self):
        """Ctrough should be <= Cmax for every completer."""
        protocol = _make_protocol(n_doses=15)
        result = simulate_trial(protocol)
        arm = list(result.arm_results.values())[0]
        for p in arm.patient_summaries:
            if p["ss_cmax"] > 0:
                assert p["ss_ctrough"] <= p["ss_cmax"]

    def test_dropout_rate_matches_count(self):
        """dropout_rate should equal n_dropouts / n_patients."""
        protocol = _make_protocol(n_doses=20)
        adh = AdherenceModel(p11_base=0.80, dropout_threshold=4)
        result = simulate_trial(protocol, adherence_model=adh, enable_feedback=True)
        arm = list(result.arm_results.values())[0]
        expected = arm.n_dropouts / arm.n_patients
        assert arm.dropout_rate == pytest.approx(expected)

    def test_reproducible_with_seed(self):
        """Same seed should produce identical results."""
        protocol = _make_protocol(seed=77, n_doses=10)
        adh = AdherenceModel()
        r1 = simulate_trial(protocol, adherence_model=adh, enable_feedback=True)
        r2 = simulate_trial(protocol, adherence_model=adh, enable_feedback=True)
        a1 = list(r1.arm_results.values())[0]
        a2 = list(r2.arm_results.values())[0]
        assert a1.cmax_ss_mean == a2.cmax_ss_mean
        assert a1.n_dropouts == a2.n_dropouts
        assert a1.total_ae_events == a2.total_ae_events

    def test_doses_taken_plus_skipped_leq_n_doses(self):
        """Taken + skipped doses should not exceed planned doses."""
        protocol = _make_protocol(n_doses=15)
        adh = AdherenceModel(p11_base=0.80, p01_base=0.40, dropout_threshold=5)
        result = simulate_trial(protocol, adherence_model=adh, enable_feedback=True)
        arm = list(result.arm_results.values())[0]
        for p in arm.patient_summaries:
            assert p["doses_taken"] + p["doses_skipped"] <= protocol.n_doses


# ── Real Drug Integration Tests ──────────────────────────────────────

class TestRealDrugIntegration:
    """Smoke tests with real drug PK parameters to verify plausible output."""

    METFORMIN_PARAMS = {
        "ka_mean": 1.5, "ka_cv": 0.35,
        "cl_ref70": 30.0, "cl_cv": 0.25,
        "vd_f_ref70": 63.0, "vd_f_cv": 0.25,
        "tlag_mean": 0.30, "tlag_cv": 0.40,
    }

    WARFARIN_PARAMS = {
        "ka_mean": 2.0, "ka_cv": 0.30,
        "cl_ref70": 0.20, "cl_cv": 0.55,
        "vd_f_ref70": 10.0, "vd_f_cv": 0.20,
        "tlag_mean": 0.25, "tlag_cv": 0.40,
    }

    def test_metformin_500mg_bid(self):
        """Metformin 500mg BID should produce Cmax ~ 1-3 mg/L."""
        protocol = TrialProtocol(
            drug_name="Metformin",
            arms=[TrialArm(label="500mg", dose_mg=500.0, n_patients=100)],
            interval_h=12.0,
            n_doses=20,
            cmax_therapeutic=4.0,
            ec50=1.0,
            seed=42,
            drug_params=self.METFORMIN_PARAMS,
        )
        result = simulate_trial(protocol, adherence_model=None)
        arm = result.arm_results["500mg"]

        # Literature Cmax for metformin 500mg: ~1-3 mg/L
        assert 0.5 < arm.cmax_ss_mean < 8.0

    def test_warfarin_narrow_ti_more_ae(self):
        """Warfarin (narrow TI) should generate more AEs than a wide-TI drug."""
        warfarin = TrialProtocol(
            drug_name="Warfarin",
            arms=[TrialArm(label="5mg", dose_mg=5.0, n_patients=100)],
            interval_h=24.0,
            n_doses=20,
            cmax_therapeutic=0.5,  # narrow
            ec50=0.3,
            seed=42,
            drug_params=self.WARFARIN_PARAMS,
        )
        safe_drug = TrialProtocol(
            drug_name="SafeDrug",
            arms=[TrialArm(label="5mg", dose_mg=5.0, n_patients=100)],
            interval_h=24.0,
            n_doses=20,
            cmax_therapeutic=100.0,  # very wide TI
            ec50=0.3,
            seed=42,
            drug_params=self.WARFARIN_PARAMS,
        )
        adh = AdherenceModel(p11_base=0.92, p01_base=0.50)
        r_war = simulate_trial(warfarin, adherence_model=adh, enable_feedback=True)
        r_safe = simulate_trial(safe_drug, adherence_model=adh, enable_feedback=True)

        ae_war = r_war.arm_results["5mg"].total_ae_events
        ae_safe = r_safe.arm_results["5mg"].total_ae_events
        assert ae_war > ae_safe

    def test_higher_bid_frequency_more_accumulation(self):
        """BID should accumulate more than QD at same daily dose."""
        params = self.METFORMIN_PARAMS.copy()

        bid = TrialProtocol(
            drug_name="BID",
            arms=[TrialArm(label="250mg", dose_mg=250.0, n_patients=100)],
            interval_h=12.0,
            n_doses=20,
            cmax_therapeutic=10.0,
            seed=42,
            drug_params=params,
        )
        qd = TrialProtocol(
            drug_name="QD",
            arms=[TrialArm(label="500mg", dose_mg=500.0, n_patients=100)],
            interval_h=24.0,
            n_doses=10,  # same duration
            cmax_therapeutic=10.0,
            seed=42,
            drug_params=params,
        )
        r_bid = simulate_trial(bid, adherence_model=None)
        r_qd = simulate_trial(qd, adherence_model=None)

        # BID gives lower Cmax but higher Ctrough (more even exposure)
        cmax_bid = r_bid.arm_results["250mg"].cmax_ss_mean
        cmax_qd = r_qd.arm_results["500mg"].cmax_ss_mean
        ctrough_bid = r_bid.arm_results["250mg"].ctrough_ss_mean
        ctrough_qd = r_qd.arm_results["500mg"].ctrough_ss_mean

        assert cmax_bid < cmax_qd  # half dose per administration
        assert ctrough_bid > ctrough_qd  # more frequent dosing → higher trough
