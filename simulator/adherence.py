"""Two-state Markov adherence model with PK-AE feedback and dose-timing jitter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simulator.patient import VirtualPatient, PatientState
from simulator.pharmacology import ae_probability


@dataclass
class AdherenceModel:
    """Two-state Markov chain with concentration-dependent transitions.

    States: ADHERENT (1), NON-ADHERENT (0)

    Transition probabilities:
        P(1->1) = p11_base * adherence_tendency - ae_penalty
        P(0->1) = p01_base + symptom_return_bonus
        P(1->0) = 1 - P(1->1)
        P(0->0) = 1 - P(0->1)

    The ae_penalty is the KEY feedback mechanism:
    high Cmax -> AE -> lower P(staying adherent) -> skip dose -> lower Cmax

    Dose-timing jitter:
        When a dose IS taken, the actual time is:
        actual_time = scheduled_time + Normal(0, timing_jitter_h)
        Clipped so doses don't overlap (>= prev_dose + min_gap_h).
    """

    p11_base: float = 0.90  # P(stay adherent | was adherent), no AE
    p01_base: float = 0.40  # P(become adherent | was non-adherent)
    dropout_threshold: int = 5  # consecutive skips -> dropout
    timing_jitter_h: float = 1.5  # SD of dose-timing noise (hours)
    min_gap_h: float = 2.0  # minimum hours between consecutive doses

    def jitter_dose_time(
        self,
        rng: np.random.Generator,
        scheduled_time: float,
        last_dose_time: float | None,
    ) -> float:
        """Apply timing jitter to a scheduled dose time.

        Returns the actual time the patient takes the dose.
        """
        if self.timing_jitter_h <= 0:
            return scheduled_time

        offset = rng.normal(0, self.timing_jitter_h)
        actual = scheduled_time + offset

        # Clip: can't take dose before minimum gap after previous dose
        if last_dose_time is not None:
            actual = max(actual, last_dose_time + self.min_gap_h)

        # Can't take dose before time 0
        actual = max(0.0, actual)

        return actual

    def decide(
        self,
        rng: np.random.Generator,
        patient: VirtualPatient,
        state: PatientState,
        last_cmax: float,
        cmax_therapeutic: float,
    ) -> str:
        """Decide action for next dose: 'take', 'skip', or 'dropout'.

        This is where the PK -> behavior feedback loop lives.
        """
        if state.dropped_out:
            return "dropout"

        # Compute AE probability from last interval's Cmax
        p_ae = ae_probability(
            last_cmax, cmax_therapeutic, patient.ae_sensitivity
        )

        # Did the patient actually experience an AE?
        experienced_ae = rng.random() < p_ae
        if experienced_ae:
            state.ae_events.append((
                state.doses_taken[-1][0] if state.doses_taken else 0,
                p_ae,
            ))

        # Modify transition probabilities based on AE
        ae_penalty = 0.3 * p_ae if experienced_ae else 0.0

        # Current adherence state
        was_adherent = state.consecutive_skips == 0

        if was_adherent:
            p_stay = max(0.1, self.p11_base * patient.adherence_tendency - ae_penalty)
            take = rng.random() < p_stay
        else:
            # Symptom return bonus: longer off -> more likely to resume
            symptom_bonus = min(0.3, 0.05 * state.consecutive_skips)
            p_resume = min(0.95, self.p01_base + symptom_bonus)
            take = rng.random() < p_resume

        if take:
            state.consecutive_skips = 0
            return "take"
        else:
            state.consecutive_skips += 1
            if state.consecutive_skips >= self.dropout_threshold:
                state.dropped_out = True
                return "dropout"
            return "skip"

    def decide_no_feedback(
        self,
        rng: np.random.Generator,
        patient: VirtualPatient,
        state: PatientState,
    ) -> str:
        """Pure Markov decision without PK-AE feedback."""
        if state.dropped_out:
            return "dropout"

        was_adherent = state.consecutive_skips == 0

        if was_adherent:
            p = self.p11_base * patient.adherence_tendency
        else:
            symptom_bonus = min(0.3, 0.05 * state.consecutive_skips)
            p = min(0.95, self.p01_base + symptom_bonus)

        if rng.random() < p:
            state.consecutive_skips = 0
            return "take"
        else:
            state.consecutive_skips += 1
            if state.consecutive_skips >= self.dropout_threshold:
                state.dropped_out = True
                return "dropout"
            return "skip"
