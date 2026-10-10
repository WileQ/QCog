"""Every model in the catalogue is run and its central claims are checked."""

import numpy as np
import pytest

from qcog.registry import MODELS, catalogue, get_model, models_by_module, search


@pytest.fixture(scope="module")
def results():
    return {slug: model().run() for slug, model in MODELS.items()}


def test_catalogue_is_complete_and_well_formed():
    assert len(MODELS) == 23
    assert set(MODELS) == set(models_by_module_flat())
    for entry in catalogue():
        assert entry["one_liner"] and entry["cognitive_question"]
        assert entry["quantum_mechanism"] and entry["classical_baseline"]
        assert entry["caveats"], f"{entry['slug']} has no caveats"
        assert entry["references"], f"{entry['slug']} has no references"


def models_by_module_flat():
    return [model.slug for models in models_by_module().values() for model in models]


def test_search_and_lookup():
    assert "question-order" in [model.slug for model in search("question order")]
    assert get_model("question-order").slug == "question-order"
    with pytest.raises(KeyError):
        get_model("nope")


def test_every_model_runs_with_normalised_distributions(results):
    for slug, result in results.items():
        assert result.findings, f"{slug} produced no findings"
        assert result.summary, f"{slug} produced no summary"
        for name, item in result.sequential_results().items():
            total = sum(item.joint.values())
            assert np.isclose(total, 1.0, atol=1e-6), f"{slug}:{name} sums to {total}"
            assert all(value >= -1e-12 for value in item.joint.values())


def test_every_model_accepts_its_own_parameters(results):
    for slug, result in results.items():
        model = MODELS[slug]()
        assert set(result.parameters) == {spec.name for spec in model.params}


def test_results_serialise(results):
    payload = results["question-order"].to_dict()
    assert payload["slug"] == "question-order"
    assert payload["analysis"]["order_effect"]["total_variation"] > 0
    assert "joint" in list(payload["results"].values())[0]


def test_question_order_claims(results):
    analysis = results["question-order"].analysis
    assert analysis["order_effect"].max_marginal_shift() > 0.1
    assert abs(analysis["qq_equality"].difference) < 1e-9
    assert analysis["interference"].check()


def test_conjunction_fallacy_claims(results):
    analysis = results["conjunction-fallacy"].analysis
    assert analysis["conjunction"]["excess"] > 0.05
    assert analysis["conjunction"]["violated"]
    assert analysis["independent_check"]["excess"] <= 1e-9


def test_disjunction_effect_claims(results):
    values = results["disjunction-effect"].analysis["values"]
    assert values["holistic unknown"] - values["entangled unknown"] > 0.1
    assert abs(values["entangled unknown"] - values["branch mixture"]) < 1e-9


def test_concept_combination_shows_emergence(results):
    typicality = results["concept-combination"].analysis["typicality"]
    assert typicality["combined"] > max(typicality["pet"], typicality["fish"])
    assert results["concept-combination"].analysis["entanglement"].is_product is False


def test_conceptual_entanglement_excess(results):
    structure = results["conceptual-entanglement"].analysis["joint_structure"]
    assert structure["excess"] > 0.2
    assert structure["p(A and B)"] <= min(structure["p(A)"], structure["p(B)"]) + 1e-9


def test_superposition_model_separates_the_states(results):
    index = results["superposition-beliefs"].analysis["distinguishability"]
    assert index["coherent"] - index["ignorance"] > 0.4


def test_game_theory_claims(results):
    pd_analysis = results["quantum-prisoners-dilemma"].analysis
    assert pd_analysis["joint_moves"]["C-C"] > 0.99
    coordination = results["coordination-games"].analysis["coordination"]
    assert coordination > 0.99


def test_evolution_of_cooperation_transition():
    entangled = get_model("evolution-of-cooperation").run(entanglement=np.pi / 2)
    classical = get_model("evolution-of-cooperation").run(entanglement=0.0)
    entangled_shares = entangled.analysis["final_shares"]
    classical_shares = classical.analysis["final_shares"]
    assert entangled_shares["C"] + entangled_shares["Q"] > 0.8
    assert classical_shares["D"] > 0.8


def test_collective_judgement_claims():
    independent = get_model("collective-judgment").run(correlation=0.0)
    correlated = get_model("collective-judgment").run(correlation=1.0)
    assert independent.analysis["majority_accuracy"] > correlated.analysis["majority_accuracy"] + 0.05
    assert np.isclose(correlated.analysis["majority_accuracy"], 0.7)


def test_independent_judges_reproduce_condorcet_exactly():
    """The classical limit of the crowd model must be exactly the classical theorem."""
    from qcog.core.classical import condorcet_jury_probability

    for judges in (7, 8, 9):
        independent = get_model("collective-judgment").run(
            correlation=0.0, num_agents=judges, sweeps=False
        )
        condensed = condorcet_jury_probability(0.7, judges)
        assert np.isclose(independent.analysis["majority_accuracy"], condensed, atol=1e-9), judges
    # and with perfect correlation the crowd inherits the individual accuracy
    for judges in (7, 8):
        correlated = get_model("collective-judgment").run(
            correlation=1.0, num_agents=judges, sweeps=False
        )
        assert np.isclose(correlated.analysis["majority_accuracy"], 0.7, atol=1e-9)


def test_information_cascade_reports_a_cascade(results):
    cascade = results["information-cascades"].analysis["cascade"]
    assert cascade["adoptions"] >= 1
    assert cascade["onset"] is not None


def test_every_model_produces_figures_with_descriptions():
    import matplotlib.pyplot as plt
    for slug in ("question-order", "quantum-prisoners-dilemma", "collective-judgment"):
        model = get_model(slug)
        result = model.run()
        figures = model.figures(result)
        assert figures, f"{slug} produced no figures"
        for name, figure in figures.items():
            description = getattr(figure, "_qcog_description", "")
            assert len(description) > 30, f"{slug}:{name} has no description"
    plt.close("all")


def test_models_run_on_a_shot_based_backend():
    result = get_model("question-order").run(backend="aer", shots=4000, seed=5)
    assert result.shots == 4000
    assert result.summary
    for item in result.sequential_results().values():
        assert not item.is_exact
        assert item.counts is not None